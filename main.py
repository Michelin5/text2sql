import asyncio
import json
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import pandas as pd
from pydantic import BaseModel
from json.decoder import JSONDecodeError
from fastapi.staticfiles import StaticFiles
import os

# Импортируем агентов и утилиты
from giga_wrapper import giga_client, call_giga_api_wrapper
from agents import (
    agent0_coordinator,
    agent1_rephraser,
    agent2_planner,
    agent3_sql_generator,
    agent_sql_validator,
    agent_sql_fixer,
    agent_anomaly_detector,
    agent_rag_retriever
)
from duckdb_utils import setup_duckdb, execute_duckdb_query
from rag_handler import init_rag_db, add_to_rag_db

# Determine the absolute path to the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "index.html")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - Use the absolute STATIC_DIR path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Root endpoint to serve index.html - THIS SHOULD BE THE ONLY @app.get("/")
@app.get("/")
async def read_index():
    # Use the absolute INDEX_HTML_PATH
    # Also, check if the file exists before trying to serve it for better error handling
    if not os.path.exists(INDEX_HTML_PATH):
        print(f"[ERROR] index.html not found at: {INDEX_HTML_PATH}")
        raise HTTPException(status_code=404, detail=f"index.html not found at {INDEX_HTML_PATH}")
    print(f"[INFO] Serving index.html from: {INDEX_HTML_PATH}")
    return FileResponse(INDEX_HTML_PATH)

# ==============================================================================
# ОБРАБОТКА НЕОПРЕДЕЛЕННОСТИ И УТОЧНЕНИЙ
# ==============================================================================

async def analyze_query_ambiguity(user_prompt):
    """Анализирует неопределенность запроса без использования координатора"""
    prompt_lower = user_prompt.lower()
    
    # Детекция аномалий
    anomaly_keywords = ['аномали', 'выброс', 'необычн', 'странн', 'отклонени', 'экстремальн', 'подозритель']
    needs_anomaly = any(keyword in prompt_lower for keyword in anomaly_keywords)
    
    # Детекция таблиц в запросе
    table_mentions = {
        'population': ['населен', 'демограф', 'людей', 'жител'],
        'salary': ['зарплат', 'заработн', 'оклад', 'доход'],
        'migration': ['мигра', 'переезд', 'перемещен'],
        'market_access': ['рынок', 'доступност', 'экономическ'],
        'connections': ['связ', 'соединен', 'маршрут']
    }
    
    mentioned_tables = []
    for table, keywords in table_mentions.items():
        if any(keyword in prompt_lower for keyword in keywords):
            mentioned_tables.append(table)
    
    # Определение уровня неопределенности
    ambiguity_level = "low"
    needs_clarification = False
    enhanced_prompt = user_prompt
    
    # Высокая неопределенность: аномалии без указания таблицы
    if (needs_anomaly and not mentioned_tables and len(user_prompt.split()) < 8):
        ambiguity_level = "high"
        needs_clarification = True
        
        clarification_data = {
            "needs_clarification": True,
            "ambiguity_level": ambiguity_level,
            "reason": "Запрос об аномалиях без указания конкретной таблицы",
            "message": "Для поиска аномалий укажите, в каких данных искать:",
            "suggested_tables": [
                {"id": "population", "name": "Население", "description": "Данные о численности населения по регионам"},
                {"id": "salary", "name": "Зарплаты", "description": "Данные о заработных платах по отраслям"},
                {"id": "migration", "name": "Миграция", "description": "Данные о миграционных потоках"},
                {"id": "market_access", "name": "Доступность рынков", "description": "Индексы доступности рынков"},
                {"id": "connections", "name": "Связи", "description": "Данные о связях между территориями"}
            ],
            "examples": [
                "Найди аномалии в данных о населении",
                "Покажи выбросы в зарплатах по отраслям",
                "Обнаружь необычные значения миграции",
                "Найди аномальные значения доступности рынков"
            ]
        }
        return clarification_data, None, needs_clarification, needs_anomaly
    
    # Средняя неопределенность: автоматическое улучшение
    elif (needs_anomaly and not mentioned_tables):
        ambiguity_level = "medium"
        enhanced_prompt = f"Найди аномалии в данных о миграции"
        
        enhancement_data = {
            "ambiguity_level": ambiguity_level,
            "original_prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt,
            "reason": "Автоматически выбрана таблица 'миграция' для поиска аномалий",
            "auto_selected_table": "migration"
        }
        return enhancement_data, enhanced_prompt, False, needs_anomaly
    
    # Низкая неопределенность
    return None, enhanced_prompt, False, needs_anomaly

# ==============================================================================
# LLM-АНАЛИЗ РЕЗУЛЬТАТОВ И ГЕНЕРАЦИЯ САММАРИ
# ==============================================================================

def generate_llm_analysis_summary(sql_results_df, user_prompt, anomaly_results=None, needs_anomaly_detection=False, rag_context: dict = None):
    """
    Генерирует LLM-анализ результатов с структурированным саммари
    """
    try:
        # Подготавливаем данные для анализа
        analysis_context = prepare_analysis_context(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)
        
        # Генерируем LLM-анализ
        llm_summary = generate_intelligent_summary(analysis_context, rag_context=rag_context)
        
        return llm_summary
    except Exception as e:
        print(f"[LLM_ANALYSIS ERROR] Ошибка в LLM-анализе: {e}")
        return generate_fallback_summary(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)

def prepare_analysis_context(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection):
    """Подготавливает контекст для LLM-анализа"""
    context = {
        "original_query": user_prompt,
        "data_available": sql_results_df is not None and not sql_results_df.empty,
        "total_records": len(sql_results_df) if sql_results_df is not None else 0,
        "analysis_type": "anomaly_detection" if needs_anomaly_detection else "standard_analysis"
    }
    
    # Добавляем информацию о данных
    if sql_results_df is not None and not sql_results_df.empty:
        context["columns"] = list(sql_results_df.columns)
        context["data_types"] = {col: str(dtype) for col, dtype in sql_results_df.dtypes.items()}
        
        # Статистика по числовым столбцам
        numeric_cols = sql_results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context["statistics"] = {}
            for col in numeric_cols:
                col_data = sql_results_df[col].dropna()
                if len(col_data) > 0:
                    context["statistics"][col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max())
                    }
        
        # Образец данных
        context["sample_data"] = sql_results_df.head(5).to_dict('records')
    
    # Добавляем результаты анализа аномалий
    if anomaly_results:
        context["anomaly_analysis"] = anomaly_results
    
    return context

def generate_intelligent_summary(analysis_context, rag_context: dict = None):
    """Генерирует интеллектуальное саммари с помощью LLM"""
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('final_answer'):
        rag_guidance = (
            "Тебе также предоставлен контекст из очень похожего успешного прошлого запроса, включая его финальный ответ. "
            "Используй стиль, тон и структуру прошлого ответа (`Контекст прошлого запроса (финальный ответ)`) как хороший пример. "
            "Однако, убедись, что ТЕКУЩИЙ ответ ТОЧНО отражает предоставленные ДАННЫЕ И АНАЛИЗ из `Контекст анализа данных`, "
            "а не данные из прошлого ответа. Адаптируй формулировки под текущие данные.\n"
            f"Контекст прошлого запроса (финальный ответ):\n{rag_context.get('final_answer', 'N/A')}\n\n"
        )

    system_instruction = f"""
Ты — эксперт-аналитик данных. На основе предоставленного контекста анализа создай структурированное, профессиональное саммари на русском языке.
{rag_guidance}
Твоя задача:
1. Проанализировать полученные данные и результаты
2. Выделить ключевые инсайты и паттерны
3. Предоставить actionable рекомендации
4. Сформулировать выводы в понятном формате

Структура ответа должна включать:
- Краткое резюме (2-3 предложения)
- Ключевые выводы (список)
- Статистические инсайты (если применимо)
- Обнаруженные аномалии (если есть)
- Рекомендации (практические советы)
- Ограничения и предостережения (если есть)

Используй профессиональный, но доступный язык. Будь конкретным и фокусируйся на практической ценности информации.
"""
    
    prompt = f"""
Контекст анализа данных:

Исходный запрос пользователя: "{analysis_context['original_query']}"

Тип анализа: {analysis_context['analysis_type']}
Данные доступны: {'Да' if analysis_context['data_available'] else 'Нет'}
Общее количество записей: {analysis_context['total_records']}

"""
    
    if analysis_context['data_available']:
        prompt += f"""
Структура данных:
- Столбцы: {', '.join(analysis_context['columns'])}
- Типы данных: {analysis_context['data_types']}

"""
        
        if 'statistics' in analysis_context:
            prompt += "Статистические показатели:\n"
            for col, stats in analysis_context['statistics'].items():
                prompt += f"- {col}: среднее={stats['mean']:.2f}, медиана={stats['median']:.2f}, σ={stats['std']:.2f}, мин={stats['min']:.2f}, макс={stats['max']:.2f}\n"
            prompt += "\n"
        
        if 'sample_data' in analysis_context:
            prompt += f"Образец данных (первые 5 записей):\n{analysis_context['sample_data']}\n\n"
    
    if 'anomaly_analysis' in analysis_context:
        anomaly_data = analysis_context['anomaly_analysis']
        prompt += f"""
Результаты анализа аномалий:
- Аномалии обнаружены: {'Да' if anomaly_data.get('anomalies_found', False) else 'Нет'}
"""
        if anomaly_data.get('anomalies_found', False):
            prompt += f"- Количество типов аномалий: {anomaly_data.get('anomaly_count', 0)}\n"
            if 'anomalies' in anomaly_data:
                prompt += "Детали аномалий:\n"
                for anomaly in anomaly_data['anomalies']:
                    if 'column' in anomaly:
                        prompt += f"  * Столбец '{anomaly['column']}': {anomaly['anomaly_count']} аномалий ({anomaly['anomaly_percentage']}%)\n"
                    elif 'territory' in anomaly:
                        prompt += f"  * Территория '{anomaly['territory']}': {anomaly['anomaly_count']} аномалий ({anomaly['anomaly_percentage']}%)\n"
        prompt += "\n"
    
    prompt += "Создай профессиональное аналитическое саммари на основе этой информации."
    
    try:
        llm_response = call_giga_api_wrapper(prompt, system_instruction)
        return format_llm_summary(llm_response, analysis_context)
    except Exception as e:
        print(f"[LLM_SUMMARY ERROR] Ошибка генерации LLM саммари: {e}")
        return generate_fallback_summary_from_context(analysis_context)

def format_llm_summary(llm_response, analysis_context):
    """Форматирует LLM ответ в структурированное саммари"""
    # Создаем заголовок
    analysis_type_name = "🔍 Анализ аномалий" if analysis_context['analysis_type'] == 'anomaly_detection' else "📊 Анализ данных"
    
    formatted_summary = f"# {analysis_type_name}\n\n"
    formatted_summary += f"**Запрос:** {analysis_context['original_query']}\n"
    formatted_summary += f"**Обработано записей:** {analysis_context['total_records']}\n\n"
    
    # Добавляем LLM анализ
    formatted_summary += "## 🤖 Интеллектуальный анализ\n\n"
    formatted_summary += llm_response + "\n\n"
    
    # Добавляем технические детали если есть аномалии
    if 'anomaly_analysis' in analysis_context and analysis_context['anomaly_analysis'].get('anomalies_found', False):
        formatted_summary += "## 📋 Технические детали аномалий\n\n"
        anomaly_data = analysis_context['anomaly_analysis']
        
        if 'anomalies' in anomaly_data:
            for i, anomaly in enumerate(anomaly_data['anomalies'], 1):
                if 'column' in anomaly:
                    formatted_summary += f"**{i}. Столбец `{anomaly['column']}`**\n"
                    formatted_summary += f"- Аномалий: {anomaly['anomaly_count']} из {anomaly['total_count']} ({anomaly['anomaly_percentage']}%)\n"
                    formatted_summary += f"- Диапазон нормы: {anomaly['bounds']['lower']:.2f} - {anomaly['bounds']['upper']:.2f}\n"
                    formatted_summary += f"- Статистика: μ={anomaly['statistics']['mean']:.2f}, med={anomaly['statistics']['median']:.2f}, σ={anomaly['statistics']['std']:.2f}\n\n"
                elif 'territory' in anomaly:
                    formatted_summary += f"**{i}. Территория: {anomaly['territory']}**\n"
                    formatted_summary += f"- Аномалий: {anomaly['anomaly_count']} из {anomaly['total_count']} ({anomaly['anomaly_percentage']}%)\n"
                    formatted_summary += f"- Max Z-score: {anomaly['statistics']['max_z_score']:.2f}\n"
                    formatted_summary += f"- Статистика: μ={anomaly['statistics']['mean']:.2f}, σ={anomaly['statistics']['std']:.2f}\n\n"
    
    # Добавляем метаданные
    formatted_summary += "---\n"
    formatted_summary += f"*Анализ выполнен системой многоагентного ИИ*\n"
    
    return formatted_summary

def generate_fallback_summary_from_context(analysis_context):
    """Генерирует резервное саммари на основе контекста"""
    analysis_type_name = "🔍 Анализ аномалий" if analysis_context['analysis_type'] == 'anomaly_detection' else "📊 Анализ данных"
    
    summary = f"# {analysis_type_name}\n\n"
    summary += f"**Запрос:** {analysis_context['original_query']}\n"
    summary += f"**Обработано записей:** {analysis_context['total_records']}\n\n"
    
    if analysis_context['data_available']:
        summary += "## 📋 Результаты анализа\n\n"
        summary += f"Успешно обработано {analysis_context['total_records']} записей данных.\n\n"
        
        if 'statistics' in analysis_context:
            summary += "**Статистические показатели:**\n"
            for col, stats in analysis_context['statistics'].items():
                summary += f"- `{col}`: среднее = {stats['mean']:.2f}, разброс = {stats['min']:.2f} - {stats['max']:.2f}\n"
            summary += "\n"
    else:
        summary += "## ⚠️ Данные не найдены\n\n"
        summary += "По указанным критериям данные не обнаружены. Рекомендуется пересмотреть параметры запроса.\n\n"
    
    if 'anomaly_analysis' in analysis_context:
        anomaly_data = analysis_context['anomaly_analysis']
        if anomaly_data.get('anomalies_found', False):
            summary += f"## 🚨 Обнаружены аномалии\n\n"
            summary += f"Найдено {anomaly_data.get('anomaly_count', 0)} типов аномалий в данных.\n\n"
        else:
            summary += f"## ✅ Аномалии не обнаружены\n\n"
            summary += "Все данные находятся в пределах нормальных значений.\n\n"
    
    return summary

def generate_fallback_summary(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection):
    """Генерирует простое резервное саммари"""
    if sql_results_df is None:
        return "**❌ Ошибка анализа**\n\nНе удалось получить данные для анализа. Проверьте корректность запроса."
    
    if sql_results_df.empty:
        return f"**📭 Данные не найдены**\n\nПо запросу '{user_prompt}' данные не обнаружены. Попробуйте изменить критерии поиска."
    
    summary = f"**📊 Краткое саммари**\n\n"
    summary += f"**Запрос:** {user_prompt}\n"
    summary += f"**Найдено записей:** {len(sql_results_df)}\n"
    summary += f"**Столбцы данных:** {', '.join(sql_results_df.columns.tolist())}\n\n"
    
    if needs_anomaly_detection and anomaly_results:
        if anomaly_results.get('anomalies_found', False):
            summary += f"**🚨 Аномалии:** Обнаружено {anomaly_results.get('anomaly_count', 0)} типов аномалий\n"
        else:
            summary += f"**✅ Аномалии:** Не обнаружены\n"
    
    return summary

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ-ГЕНЕРАТОР (ОБНОВЛЕННАЯ)
# ==============================================================================

async def event_generator(user_prompt: str, duckdb_con):
    """
    Асинхронный генератор событий для обработки запроса пользователя.
    Включает RAG, агентов, выполнение SQL и генерацию ответа.
    """
    current_stage = "Начало обработки"
    current_data = {}
    rag_context_for_agents = None
    
    # Accumulated data for potential saving
    accumulated_data_for_rag = {
        "user_prompt": user_prompt,
        "formal_prompt": None,
        "plan": None,
        "sql_query": None,
        "final_answer": None
    }

    try:
        yield {"type": "status", "stage": "rag_check", "message": "Проверка базы знаний похожих запросов..."}
        rag_result = agent_rag_retriever(user_prompt)
        rag_status = rag_result.get("status")
        rag_data = rag_result.get("data")

        if rag_status == "error":
            yield {"type": "error", "stage": "rag_check", "message": f"Ошибка при доступе к базе знаний: {rag_result.get('message')}. Продолжаем без нее."}
        elif rag_status == "exact":
            final_answer = rag_data.get("final_answer", "Точный ответ найден, но не удалось извлечь его содержимое.")
            accumulated_data_for_rag["formal_prompt"] = rag_data.get("formal_prompt")
            accumulated_data_for_rag["plan"] = rag_data.get("plan")
            accumulated_data_for_rag["sql_query"] = rag_data.get("sql_query")
            accumulated_data_for_rag["final_answer"] = final_answer
            yield {"type": "final_answer", "stage": "rag_exact_match", "answer": final_answer, "rag_components": accumulated_data_for_rag, "message": "Найден точный ответ в базе знаний."}
            print(f"[EVENT_GENERATOR] RAG exact match. Returning stored answer for: {user_prompt[:50]}")
            return
        elif rag_status == "similar":
            rag_context_for_agents = rag_data
            yield {"type": "status", "stage": "rag_similar_match", "message": "Найден похожий запрос в базе знаний. Используем его для улучшения ответа.", "similar_data_preview": {k: (v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) for k,v in rag_data.items()}}
            print(f"[EVENT_GENERATOR] RAG similar match found for: {user_prompt[:50]}")
        else:
            yield {"type": "status", "stage": "rag_no_match", "message": "Похожих запросов в базе знаний не найдено. Стандартная обработка."}
            print(f"[EVENT_GENERATOR] RAG no match for: {user_prompt[:50]}")

        # 0. Координатор (Anomaly detection check)
        current_stage = "Определение типа анализа (координатор)"
        yield {"type": "status", "stage": "coordinator", "message": current_stage + "..."}
        ambiguity_analysis_result, enhanced_prompt, needs_clarification, needs_anomaly_detection_local = await analyze_query_ambiguity(user_prompt)
        
        if needs_clarification:
            yield {"type": "clarification_needed", "stage": "ambiguity_analysis", "data": ambiguity_analysis_result}
            print(f"[EVENT_GENERATOR] Clarification needed for: {user_prompt[:50]}")
            return

        if enhanced_prompt and enhanced_prompt != user_prompt:
            yield {"type": "status", "stage": "ambiguity_analysis", "message": f"Запрос был автоматически улучшен: {enhanced_prompt}", "original_prompt": user_prompt, "enhanced_prompt": enhanced_prompt}
            processed_user_prompt = enhanced_prompt
        else:
            processed_user_prompt = user_prompt
        
        needs_anomaly_detection = needs_anomaly_detection_local

        # 1. Переформулировщик
        current_stage = "Переформулировка запроса"
        yield {"type": "status", "stage": "rephraser", "message": current_stage + "..."}
        formal_prompt_result = agent1_rephraser(processed_user_prompt, rag_context=rag_context_for_agents)
        accumulated_data_for_rag["formal_prompt"] = formal_prompt_result
        yield {"type": "intermediary_result", "stage": "rephraser", "name": "Формальный запрос", "content": formal_prompt_result}

        # 2. Планировщик
        current_stage = "Создание плана"
        yield {"type": "status", "stage": "planner", "message": current_stage + "..."}
        plan_result = agent2_planner(formal_prompt_result, rag_context=rag_context_for_agents)
        accumulated_data_for_rag["plan"] = plan_result
        yield {"type": "intermediary_result", "stage": "planner", "name": "План выполнения", "content": plan_result}

        # 3. SQL Генератор
        current_stage = "Генерация SQL"
        yield {"type": "status", "stage": "sql_generator", "message": current_stage + "..."}
        sql_query_result = agent3_sql_generator(plan_result, rag_context=rag_context_for_agents)
        yield {"type": "intermediary_result", "stage": "sql_generator", "name": "Сгенерированный SQL", "content": sql_query_result, "language": "sql"}

        # 4. Валидация и исправление SQL
        current_stage = "Валидация SQL"
        max_fix_attempts = 3
        current_fix_attempt = 0
        validated_sql = None

        for attempt in range(max_fix_attempts):
            current_fix_attempt = attempt + 1
            yield {"type": "status", "stage": "sql_validation", "message": f"{current_stage} (попытка {current_fix_attempt}/{max_fix_attempts})..."}
            validator_result = agent_sql_validator(sql_query_result, plan_result)

            if validator_result.get("is_valid"):
                validated_sql = validator_result.get("corrected_sql") or sql_query_result
                yield {"type": "intermediary_result", "stage": "sql_validation", "name": "Валидированный SQL", "content": validated_sql, "language": "sql", "message": "SQL прошел валидацию."}
                break
            else:
                error_message = validator_result.get("message", "SQL не прошел валидацию.")
                yield {"type": "warning", "stage": "sql_validation", "message": f"Ошибка валидации SQL: {error_message}"}
                
                if validator_result.get("corrected_sql"):
                    sql_query_result = validator_result["corrected_sql"]
                    yield {"type": "status", "stage": "sql_auto_correction", "message": "Валидатор предложил исправление. Повторная валидация...", "corrected_sql": sql_query_result}
                    continue

                current_stage = "Исправление SQL (агентом)"
                yield {"type": "status", "stage": "sql_fixer", "message": current_stage + "..."}
                fixed_sql = agent_sql_fixer(sql_query_result, error_message, plan_result)
                if fixed_sql and fixed_sql.strip():
                    sql_query_result = fixed_sql
                    yield {"type": "intermediary_result", "stage": "sql_fixer", "name": "Исправленный SQL (агентом)", "content": sql_query_result, "language": "sql"}
                    current_stage = "Валидация SQL"
                else:
                    yield {"type": "error", "stage": "sql_fixer", "message": "Агент не смог исправить SQL. Пожалуйста, проверьте план или запрос."}
                    return

        if not validated_sql:
            yield {"type": "error", "stage": "sql_validation", "message": "Не удалось получить валидный SQL после нескольких попыток."}
            return
        
        accumulated_data_for_rag["sql_query"] = validated_sql

        # 5. Выполнение SQL
        current_stage = "Выполнение SQL в DuckDB"
        yield {"type": "status", "stage": "sql_execution", "message": current_stage + "..."}
        try:
            sql_results_df = await execute_duckdb_query(duckdb_con, validated_sql)
            if sql_results_df is not None and not sql_results_df.empty:
                for col in sql_results_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sql_results_df[col]):
                        sql_results_df[col] = sql_results_df[col].astype(str)
                sql_results_json = sql_results_df.to_dict(orient='records')
                print(f"[EVENT_GENERATOR] SQL executed successfully. Rows returned: {len(sql_results_json)}")
            else:
                sql_results_json = []
                print(f"[EVENT_GENERATOR] SQL executed successfully but returned no data. Query was:\n{validated_sql}")
            yield {"type": "sql_result", "stage": "sql_execution", "data": sql_results_json, "row_count": len(sql_results_json), "message": "SQL выполнен успешно."}
        except Exception as e:
            yield {"type": "error", "stage": "sql_execution", "message": f"Ошибка выполнения SQL: {str(e)}"}
            print(f"[EVENT_GENERATOR ERROR] SQL execution failed. Query was:\n{validated_sql}\nError: {str(e)}")
            current_stage = "Исправление SQL после ошибки БД"
            yield {"type": "status", "stage": "sql_fixer_db_error", "message": current_stage + "..."}
            fixed_sql_db_error = agent_sql_fixer(validated_sql, str(e), plan_result)
            if fixed_sql_db_error and fixed_sql_db_error.strip():
                validated_sql = fixed_sql_db_error
                accumulated_data_for_rag["sql_query"] = validated_sql
                yield {"type": "intermediary_result", "stage": "sql_fixer_db_error", "name": "Исправленный SQL (после ошибки БД)", "content": validated_sql, "language": "sql"}
                yield {"type": "status", "stage": "sql_execution_retry", "message": "Повторное выполнение исправленного SQL..."}
                try:
                    sql_results_df = await execute_duckdb_query(duckdb_con, validated_sql)
                    if sql_results_df is not None and not sql_results_df.empty:
                        for col in sql_results_df.columns:
                            if pd.api.types.is_datetime64_any_dtype(sql_results_df[col]):
                                sql_results_df[col] = sql_results_df[col].astype(str)
                        sql_results_json = sql_results_df.to_dict(orient='records')
                    else:
                        sql_results_json = []
                    yield {"type": "sql_result", "stage": "sql_execution_retry", "data": sql_results_json, "row_count": len(sql_results_json), "message": "Исправленный SQL выполнен успешно."}
                except Exception as e_retry:
                    yield {"type": "error", "stage": "sql_execution_retry", "message": f"Ошибка повторного выполнения SQL: {str(e_retry)}"}
                    return
            else:
                yield {"type": "error", "stage": "sql_fixer_db_error", "message": "Агент не смог исправить SQL после ошибки БД."}
                return

        # 6. Анализ аномалий (если нужно)
        anomaly_summary = None
        if needs_anomaly_detection:
            current_stage = "Анализ аномалий"
            yield {"type": "status", "stage": "anomaly_detection", "message": current_stage + "..."}
            if sql_results_df is not None and not sql_results_df.empty:
                anomaly_results = agent_anomaly_detector(sql_results_df.copy(), processed_user_prompt)
                yield {"type": "intermediary_result", "stage": "anomaly_detection", "name": "Результаты анализа аномалий", "content": anomaly_results}
                anomaly_summary = anomaly_results
            else:
                 yield {"type": "warning", "stage": "anomaly_detection", "message": "Нет данных для анализа аномалий."}

        # 7. Генерация LLM саммари / ответа
        current_stage = "Генерация итогового ответа"
        yield {"type": "status", "stage": "final_summary_generation", "message": current_stage + "..."}
        llm_summary_result = generate_llm_analysis_summary(
                sql_results_df, 
            processed_user_prompt, 
            anomaly_results=anomaly_summary,
            needs_anomaly_detection=needs_anomaly_detection,
            rag_context=rag_context_for_agents
        )
        accumulated_data_for_rag["final_answer"] = llm_summary_result
        yield {"type": "final_answer", "stage": "final_summary_generation", "answer": llm_summary_result, "rag_components": accumulated_data_for_rag, "message": "Итоговый ответ сгенерирован."}

    except Exception as e:
        error_message = f"Критическая ошибка в процессе обработки запроса на этапе '{current_stage}': {str(e)}"
        print(f"[EVENT_GENERATOR ERROR] {error_message}")
        import traceback
        traceback.print_exc()
        yield {"type": "error", "stage": current_stage, "message": error_message}
    finally:
        yield {"type": "status", "stage": "done", "message": "Обработка запроса завершена."}

# ==============================================================================
# ЭНДПОИНТЫ (БЕЗ ИЗМЕНЕНИЙ)
# ==============================================================================

@app.get("/process_query_stream")
async def process_query_stream(user_prompt: str, request: Request):
    """Endpoint для обработки запросов с потоковым ответом"""
    duckdb_con = request.app.state.duckdb_con
    if not duckdb_con:
        return JSONResponse(
            status_code=500, 
            content={"error": "База данных не инициализирована."}
        )
    
    if not user_prompt or not user_prompt.strip():
        return JSONResponse(
            status_code=400, 
            content={"error": "user_prompt не предоставлен или пуст."}
        )

    print(f"\n[API] Получен запрос: {user_prompt}")

    async def format_sse(data: dict) -> str:
        """Format data as SSE message"""
        return f"data: {json.dumps(data)}\n\n"

    async def event_stream():
        async for event in event_generator(user_prompt.strip(), duckdb_con):
            yield await format_sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        }
    )

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    try:
        if hasattr(app.state, 'duckdb_con') and app.state.duckdb_con:
            return {"status": "healthy", "service": "text2sql-multi-agent", "database": "connected"}
        else:
            return {"status": "unhealthy", "service": "text2sql-multi-agent", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "service": "text2sql-multi-agent", "error": str(e)}

@app.get("/tables")
async def get_available_tables():
    """Возвращает список доступных таблиц"""
    return {
        "tables": [
            {"name": "population", "description": "Данные о численности населения по регионам", "type": "demographic"},
            {"name": "salary", "description": "Данные о заработных платах по отраслям", "type": "economic"},
            {"name": "migration", "description": "Данные о миграционных потоках", "type": "demographic"},
            {"name": "market_access", "description": "Индексы доступности рынков", "type": "economic"},
            {"name": "connections", "description": "Данные о связях между территориями", "type": "infrastructure"}
        ]
    }

# ==============================================================================
# EXCEPTION HANDLERS И STARTUP/SHUTDOWN (БЕЗ ИЗМЕНЕНИЙ)
# ==============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений"""
    print(f"[GLOBAL ERROR] Необработанная ошибка: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "message": "Произошла неожиданная ошибка. Пожалуйста, попробуйте еще раз.",
            "type": "internal_server_error"
        }
    )

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    print("🚀 Запуск Text2SQL Multi-Agent System with LLM Analysis...")
    
    try:
        duckdb_connection = setup_duckdb()
        if duckdb_connection is not None:
            app.state.duckdb_con = duckdb_connection
            print("✅ База данных DuckDB инициализирована")
        else:
            print("❌ Не удалось инициализировать базу данных")
            app.state.duckdb_con = None

        # Initialize RAG database
        print("Initializing RAG database for successful queries...")
        if not init_rag_db():
            print("[STARTUP ERROR] Failed to initialize RAG database. Application might not work as expected with RAG features.")
        else:
            print("RAG database initialized successfully.")
    except Exception as e:
        print(f"❌ Ошибка инициализации базы данных: {e}")
        app.state.duckdb_con = None

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при завершении"""
    print("🛑 Завершение работы Text2SQL Multi-Agent System...")
    
    try:
        if (hasattr(app.state, 'duckdb_con') and 
            app.state.duckdb_con is not None and
            hasattr(app.state.duckdb_con, 'close')):
            app.state.duckdb_con.close()
            print("✅ Соединение с базой данных закрыто")
        else:
            print("ℹ️ Соединение с базой данных уже закрыто или не инициализировано")
    except Exception as e:
        print(f"⚠️ Ошибка при закрытии соединения: {e}")

# ==============================================================================
# ЗАПУСК СЕРВЕРА
# ==============================================================================

# Move RagSaveRequest definition here, before the endpoint that uses it.
class RagSaveRequest(BaseModel):
    user_prompt: str
    formal_prompt: str
    plan: str
    sql_query: str
    final_answer: str

@app.post("/llm_query/confirm_and_save")
async def confirm_answer_and_save_to_rag(payload: RagSaveRequest):
    """
    Endpoint to confirm a user liked an answer and save it to the RAG database.
    """
    print(f"[API /confirm_and_save] Received request to save: {payload.user_prompt[:50]}...")
    try:
        # Run the synchronous add_to_rag_db in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, lambda: add_to_rag_db(
            user_prompt=payload.user_prompt,
            formal_prompt=payload.formal_prompt,
            plan=payload.plan,
            sql_query=payload.sql_query,
            final_answer=payload.final_answer
        ))

        if success:
            print(f"[API /confirm_and_save] Successfully saved to RAG: {payload.user_prompt[:50]}")
            return JSONResponse(content={"status": "success", "message": "Answer saved to RAG successfully."}, status_code=200)
        else:
            print(f"[API /confirm_and_save] Failed to save to RAG: {payload.user_prompt[:50]}")
            raise HTTPException(status_code=500, detail="Failed to save answer to RAG database.")
    except Exception as e:
        print(f"[API /confirm_and_save] Error during RAG save: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while saving to RAG: {str(e)}")

if __name__ == "__main__":
    print("🚀 Запуск Text2SQL Multi-Agent System with LLM Analysis...")
    # Ensure the static directory and index.html are checked at startup for debugging
    if not os.path.isdir(STATIC_DIR):
        print(f"[CRITICAL ERROR] Static directory not found: {STATIC_DIR}")
    elif not os.path.exists(INDEX_HTML_PATH):
        print(f"[CRITICAL ERROR] index.html not found in static directory: {INDEX_HTML_PATH}")
    
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5002,
        reload=False, # Set to True for development if you want auto-reloading on code changes
        log_level="info",
        access_log=True
    )
