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

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Root endpoint to serve index.html
@app.get("/")
async def read_index():
    if not os.path.exists(INDEX_HTML_PATH):
        print(f"[ERROR] index.html not found at: {INDEX_HTML_PATH}")
        raise HTTPException(status_code=404, detail=f"index.html not found at {INDEX_HTML_PATH}")
    print(f"[INFO] Serving index.html from: {INDEX_HTML_PATH}")
    return FileResponse(INDEX_HTML_PATH)

# ==============================================================================
# ОБРАБОТКА НЕОПРЕДЕЛЕННОСТИ И УТОЧНЕНИЙ (ОПТИМИЗИРОВАННАЯ)
# ==============================================================================

async def analyze_query_ambiguity(user_prompt):
    """Анализирует неопределенность запроса с более гибкой логикой"""
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
    
    # УЛУЧШЕННАЯ ЛОГИКА: Более агрессивное автоматическое улучшение
    
    # 1. Если есть аномалии без таблицы - автоматически выбираем таблицу
    if needs_anomaly and not mentioned_tables:
        enhanced_prompt = f"Найди аномалии в данных о населении"
        
        enhancement_data = {
            "ambiguity_level": "medium",
            "original_prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt,
            "reason": "Автоматически выбрана таблица 'население' для поиска аномалий",
            "auto_selected_table": "population"
        }
        return enhancement_data, enhanced_prompt, False, needs_anomaly
    
    # 2. Если запрос о топе/рейтинге без таблицы - выбираем salary
    elif any(word in prompt_lower for word in ['топ', 'лучш', 'высок', 'максимальн', 'рейтинг']):
        if not mentioned_tables:
            enhanced_prompt = f"Покажи топ 5 регионов по зарплатам за 2023 год"
            
            enhancement_data = {
                "ambiguity_level": "medium", 
                "original_prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "reason": "Автоматически выбрана таблица 'зарплаты' для рейтинга",
                "auto_selected_table": "salary"
            }
            return enhancement_data, enhanced_prompt, False, needs_anomaly
    
    # 3. Если общий запрос о данных - выбираем население
    elif any(word in prompt_lower for word in ['данные', 'информация', 'покажи', 'анализ']) and len(user_prompt.split()) < 6:
        enhanced_prompt = f"Покажи общую статистику по населению за последний год"
        
        enhancement_data = {
            "ambiguity_level": "medium",
            "original_prompt": user_prompt, 
            "enhanced_prompt": enhanced_prompt,
            "reason": "Автоматически выбран анализ данных о населении",
            "auto_selected_table": "population"
        }
        return enhancement_data, enhanced_prompt, False, needs_anomaly
    
    # 4. ТОЛЬКО для экстремально коротких запросов требуем уточнение
    if len(user_prompt.split()) <= 2 and user_prompt.lower() in ['данные', 'анализ', 'информация', 'покажи']:
        clarification_data = {
            "needs_clarification": True,
            "ambiguity_level": "high",
            "reason": "Слишком короткий запрос требует уточнения",
            "message": "Уточните, какой анализ вас интересует:",
            "suggested_tables": [
                {"id": "population", "name": "Население", "description": "Данные о численности населения"},
                {"id": "salary", "name": "Зарплаты", "description": "Данные о заработных платах"},
                {"id": "migration", "name": "Миграция", "description": "Данные о миграционных потоках"}
            ],
            "examples": [
                "Найди аномалии в данных о населении",
                "Покажи топ 5 регионов по зарплатам",
                "Анализ миграционных потоков за 2023 год"
            ]
        }
        return clarification_data, None, True, needs_anomaly
    
    # Во всех остальных случаях - обрабатываем как есть
    return None, user_prompt, False, needs_anomaly

# ==============================================================================
# ОПТИМИЗИРОВАННАЯ LLM-АНАЛИЗ С ВНУТРЕННЕЙ ЦЕПЬЮ РАССУЖДЕНИЙ
# ==============================================================================

def generate_llm_analysis_summary(sql_results_df, user_prompt, anomaly_results=None, needs_anomaly_detection=False, rag_context: dict = None):
    """Генерирует компактный LLM-анализ с внутренней цепью рассуждений"""
    try:
        # Подготавливаем компактный контекст
        analysis_context = prepare_compact_analysis_context(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)
        
        # Генерируем компактный LLM-анализ с внутренней цепью рассуждений
        llm_summary = generate_compact_intelligent_summary(analysis_context, rag_context=rag_context)
        
        return llm_summary
    except Exception as e:
        print(f"[LLM_ANALYSIS ERROR] Ошибка в компактном LLM-анализе: {e}")
        return generate_compact_fallback_summary(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)

def prepare_compact_analysis_context(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection):
    """Подготавливает компактный контекст для оптимизированного анализа"""
    context = {
        "original_query": user_prompt,
        "data_available": sql_results_df is not None and not sql_results_df.empty,
        "total_records": len(sql_results_df) if sql_results_df is not None else 0,
        "analysis_type": "anomaly_detection" if needs_anomaly_detection else "standard_analysis"
    }
    
    # ОГРАНИЧЕННАЯ информация о данных для компактности
    if sql_results_df is not None and not sql_results_df.empty:
        context["columns"] = list(sql_results_df.columns)[:5]  # Только первые 5 столбцов
        
        # Статистика только по первому числовому столбцу
        numeric_cols = sql_results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            key_col = numeric_cols[0]
            col_data = sql_results_df[key_col].dropna()
            if len(col_data) > 0:
                context["key_statistics"] = {
                    "column": key_col,
                    "mean": round(float(col_data.mean()), 2),
                    "min": round(float(col_data.min()), 2),
                    "max": round(float(col_data.max()), 2),
                    "std": round(float(col_data.std()), 2)
                }
        
        # ПЕРВЫЕ 10 СТРОК данных для показа
        context["sample_data"] = sql_results_df.head(10).to_dict('records')
    
    # СЖАТЫЕ результаты анализа аномалий - ТОЛЬКО ТОП-10
    if anomaly_results:
        compact_anomaly_results = {}
        if anomaly_results.get('anomalies_found', False):
            compact_anomaly_results["found"] = True
            compact_anomaly_results["summary"] = anomaly_results.get('description', 'Обнаружены аномалии')
            
            # ОГРАНИЧИВАЕМ ДО 10 АНОМАЛИЙ
            if 'anomalies' in anomaly_results:
                limited_anomalies = anomaly_results['anomalies'][:10]
                compact_anomaly_results["details"] = []
                
                for anomaly in limited_anomalies:
                    compact_detail = {}
                    if 'territory' in anomaly:
                        compact_detail["name"] = anomaly['territory']
                        compact_detail["type"] = "territory"
                    elif 'column' in anomaly:
                        compact_detail["name"] = anomaly['column']
                        compact_detail["type"] = "column"
                    
                    compact_detail["count"] = anomaly.get('anomaly_count', 0)
                    compact_detail["percentage"] = anomaly.get('anomaly_percentage', 0)
                    
                    # ТОЛЬКО ПЕРВЫЕ 3 ПРИМЕРА
                    if 'outlier_values' in anomaly:
                        compact_detail["examples"] = [round(x, 2) for x in anomaly['outlier_values'][:3]]
                    
                    compact_anomaly_results["details"].append(compact_detail)
        else:
            compact_anomaly_results["found"] = False
            compact_anomaly_results["message"] = anomaly_results.get('message', 'Аномалии не найдены')
        
        context["anomaly_analysis"] = compact_anomaly_results
    
    return context

def generate_compact_intelligent_summary(analysis_context, rag_context: dict = None):
    """Генерирует компактное интеллектуальное саммари с внутренней цепью рассуждений"""
    
    # ВНУТРЕННЯЯ ЦЕПЬ РАССУЖДЕНИЙ (НЕ ПОКАЗЫВАЕМ ПОЛЬЗОВАТЕЛЮ)
    internal_reasoning = f"""
    Шаг 1: Анализ запроса - "{analysis_context['original_query']}"
    Шаг 2: Тип анализа - {analysis_context['analysis_type']}
    Шаг 3: Объем данных - {analysis_context['total_records']} записей
    Шаг 4: Аномалии - {"найдены" if analysis_context.get('anomaly_analysis', {}).get('found', False) else "не найдены"}
    Шаг 5: Структура ответа - компактное саммари с примерами
    """
    
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('final_answer'):
        rag_guidance = (
            "Тебе также предоставлен контекст из похожего успешного запроса. "
            "Используй его стиль и структуру как пример, но ТОЧНО отражай ТЕКУЩИЕ данные.\n"
            f"Пример из прошлого запроса:\n{rag_context.get('final_answer', 'N/A')[:200]}...\n\n"
        )

    system_instruction = f"""
Ты — эксперт-аналитик данных. Создай КОМПАКТНОЕ саммари на русском языке (максимум 200 слов).

ВНУТРЕННЯЯ ЦЕПЬ РАССУЖДЕНИЙ (НЕ показывай пользователю):
{internal_reasoning}

{rag_guidance}

ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ответа:
1. **Краткое резюме** (1-2 предложения)
2. **Образец данных** (первые 10-15 строк в таблице)
3. **Характеристика данных** (основные показатели)
4. **Аномалии** (конкретные примеры если есть, или "не найдены")
5. **Итог** (1 предложение)

Будь конкретным, используй числа, примеры. НЕ показывай процесс рассуждений.
"""
    
    prompt = f"""
Исходный запрос: "{analysis_context['original_query']}"
Тип анализа: {analysis_context['analysis_type']}
Записей обработано: {analysis_context['total_records']}
Данные доступны: {'Да' if analysis_context['data_available'] else 'Нет'}

"""
    
    if analysis_context['data_available']:
        if 'key_statistics' in analysis_context:
            stats = analysis_context['key_statistics']
            prompt += f"""
Основной показатель: {stats['column']}
Статистика: среднее={stats['mean']}, мин={stats['min']}, макс={stats['max']}, σ={stats['std']}

Первые 10 строк данных для показа:
{analysis_context.get('sample_data', [])}

"""
    
    if 'anomaly_analysis' in analysis_context:
        anomaly_data = analysis_context['anomaly_analysis']
        if anomaly_data.get('found', False):
            prompt += f"""
АНОМАЛИИ НАЙДЕНЫ:
Краткое описание: {anomaly_data.get('summary', 'Обнаружены аномалии')}

Детали (топ-10):
"""
            for detail in anomaly_data.get('details', [])[:10]:
                prompt += f"- {detail['name']}: {detail['count']} аномалий ({detail['percentage']}%)"
                if 'examples' in detail:
                    prompt += f", примеры: {detail['examples']}"
                prompt += "\n"
        else:
            prompt += f"АНОМАЛИИ НЕ НАЙДЕНЫ: {anomaly_data.get('message', 'Все данные в норме')}\n"
    
    prompt += "\nСоздай компактное структурированное саммари:"
    
    try:
        llm_response = call_giga_api_wrapper(prompt, system_instruction)
        return format_compact_llm_summary(llm_response, analysis_context)
    except Exception as e:
        print(f"[COMPACT_LLM_SUMMARY ERROR] Ошибка генерации: {e}")
        return generate_compact_fallback_from_context(analysis_context)

def format_compact_llm_summary(llm_response, analysis_context):
    """Форматирует компактное LLM саммари"""
    analysis_type_name = "🔍 Анализ аномалий" if analysis_context['analysis_type'] == 'anomaly_detection' else "📊 Анализ данных"
    
    formatted_summary = f"# {analysis_type_name}\n\n"
    formatted_summary += f"**Запрос:** {analysis_context['original_query']}\n"
    formatted_summary += f"**Обработано:** {analysis_context['total_records']:,} записей\n\n"
    
    # LLM анализ
    formatted_summary += llm_response + "\n\n"
    
    # ПЕРВЫЕ 10-15 СТРОК ДАННЫХ (если есть)
    if analysis_context['data_available'] and 'sample_data' in analysis_context:
        formatted_summary += "**📋 Образец данных (первые 10 строк):**\n\n"
        sample_df = pd.DataFrame(analysis_context['sample_data'])
        formatted_summary += sample_df.to_markdown(index=False) + "\n\n"
    
    # КОМПАКТНЫЕ АНОМАЛИИ
    if 'anomaly_analysis' in analysis_context and analysis_context['anomaly_analysis'].get('found', False):
        anomaly_data = analysis_context['anomaly_analysis']
        formatted_summary += "**🎯 Конкретные аномалии (топ-10):**\n"
        
        for i, detail in enumerate(anomaly_data.get('details', [])[:10], 1):
            name = detail.get('name', f'Объект {i}')
            count = detail.get('count', 0)
            formatted_summary += f"{i}. **{name}**: {count} аномалий"
            
            if 'examples' in detail:
                examples_str = ", ".join([str(x) for x in detail['examples'][:3]])
                formatted_summary += f" (примеры: {examples_str})"
            formatted_summary += "\n"
        
        formatted_summary += "\n"
    
    # КРАТКИЙ ИТОГ
    formatted_summary += f"**📊 Итог:** {analysis_context['total_records']:,} записей"
    if 'anomaly_analysis' in analysis_context and analysis_context['anomaly_analysis'].get('found', False):
        total_anomalies = sum(d.get('count', 0) for d in analysis_context['anomaly_analysis'].get('details', []))
        formatted_summary += f" • {total_anomalies} аномалий найдено"
    else:
        formatted_summary += " • аномалии не обнаружены"
    
    return formatted_summary

def generate_compact_fallback_from_context(analysis_context):
    """Генерирует компактное резервное саммари"""
    analysis_type_name = "🔍 Анализ аномалий" if analysis_context['analysis_type'] == 'anomaly_detection' else "📊 Анализ данных"
    
    summary = f"# {analysis_type_name}\n\n"
    summary += f"**Запрос:** {analysis_context['original_query']}\n"
    summary += f"**Результат:** Обработано {analysis_context['total_records']} записей\n\n"
    
    if analysis_context['data_available']:
        if 'key_statistics' in analysis_context:
            stats = analysis_context['key_statistics']
            summary += f"**Основные показатели:**\n"
            summary += f"• Столбец: {stats['column']}\n"
            summary += f"• Среднее: {stats['mean']}\n"
            summary += f"• Диапазон: {stats['min']} — {stats['max']}\n\n"
        
        # Показываем образец данных
        if 'sample_data' in analysis_context:
            summary += "**📋 Образец данных:**\n\n"
            sample_df = pd.DataFrame(analysis_context['sample_data'])
            summary += sample_df.head(10).to_markdown(index=False) + "\n\n"
    else:
        summary += "**⚠️ Данные не найдены**\n\n"
    
    if 'anomaly_analysis' in analysis_context:
        if analysis_context['anomaly_analysis'].get('found', False):
            summary += "**🚨 Найдены аномалии** - проверьте данные\n"
        else:
            summary += "**✅ Аномалии не найдены** - данные в норме\n"
    
    return summary

def generate_compact_fallback_summary(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection):
    """Генерирует простое компактное резервное саммари"""
    if sql_results_df is None:
        return "**❌ Ошибка анализа**\n\nНе удалось получить данные. Проверьте корректность запроса."
    
    if sql_results_df.empty:
        return f"**📭 Данные не найдены**\n\nПо запросу '{user_prompt}' данные не обнаружены."
    
    summary = f"**📊 Краткий анализ**\n\n"
    summary += f"**Запрос:** {user_prompt}\n"
    summary += f"**Записей:** {len(sql_results_df):,}\n\n"
    
    # Показываем первые 10 строк
    summary += "**📋 Данные (первые 10 строк):**\n\n"
    summary += sql_results_df.head(10).to_markdown(index=False) + "\n\n"
    
    if needs_anomaly_detection and anomaly_results:
        if anomaly_results.get('anomalies_found', False):
            summary += "**🚨 Аномалии найдены**\n"
        else:
            summary += "**✅ Аномалии не найдены**\n"
    
    return summary

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ-ГЕНЕРАТОР (ОПТИМИЗИРОВАННАЯ)
# ==============================================================================

async def event_generator(user_prompt: str, duckdb_con):
    """Оптимизированный асинхронный генератор событий"""
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
        yield {"type": "status", "stage": "rag_check", "message": "Проверка базы знаний..."}
        rag_result = agent_rag_retriever(user_prompt)
        rag_status = rag_result.get("status")
        rag_data = rag_result.get("data")

        if rag_status == "error":
            yield {"type": "error", "stage": "rag_check", "message": f"Ошибка базы знаний: {rag_result.get('message')}. Продолжаем без нее."}
        elif rag_status == "exact":
            final_answer = rag_data.get("final_answer", "Точный ответ найден.")
            accumulated_data_for_rag["formal_prompt"] = rag_data.get("formal_prompt")
            accumulated_data_for_rag["plan"] = rag_data.get("plan")
            accumulated_data_for_rag["sql_query"] = rag_data.get("sql_query")
            accumulated_data_for_rag["final_answer"] = final_answer
            yield {"type": "final_answer", "stage": "rag_exact_match", "answer": final_answer, "rag_components": accumulated_data_for_rag, "message": "Найден точный ответ."}
            return
        elif rag_status == "similar":
            rag_context_for_agents = rag_data
            yield {"type": "status", "stage": "rag_similar_match", "message": "Найден похожий запрос. Используем для улучшения."}
        else:
            yield {"type": "status", "stage": "rag_no_match", "message": "Стандартная обработка."}

        # Анализ неопределенности
        current_stage = "Анализ запроса"
        yield {"type": "status", "stage": "coordinator", "message": current_stage + "..."}
        ambiguity_analysis_result, enhanced_prompt, needs_clarification, needs_anomaly_detection_local = await analyze_query_ambiguity(user_prompt)
        
        if needs_clarification:
            yield {"type": "clarification_needed", "stage": "ambiguity_analysis", "data": ambiguity_analysis_result}
            return

        if enhanced_prompt and enhanced_prompt != user_prompt:
            yield {"type": "status", "stage": "ambiguity_analysis", "message": f"Запрос улучшен: {enhanced_prompt}", "original_prompt": user_prompt, "enhanced_prompt": enhanced_prompt}
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
        yield {"type": "intermediary_result", "stage": "sql_generator", "name": "SQL", "content": sql_query_result, "language": "sql"}

        # 4. Валидация SQL (упрощенная)
        current_stage = "Валидация SQL"
        yield {"type": "status", "stage": "sql_validation", "message": current_stage + "..."}
        validator_result = agent_sql_validator(sql_query_result, plan_result)

        if validator_result.get("is_valid"):
            validated_sql = validator_result.get("corrected_sql") or sql_query_result
            yield {"type": "intermediary_result", "stage": "sql_validation", "name": "Валидированный SQL", "content": validated_sql, "language": "sql"}
        else:
            # Простое исправление
            current_stage = "Исправление SQL"
            yield {"type": "status", "stage": "sql_fixer", "message": current_stage + "..."}
            fixed_sql = agent_sql_fixer(sql_query_result, validator_result.get("message", ""), plan_result)
            validated_sql = fixed_sql if fixed_sql and fixed_sql.strip() else sql_query_result
            yield {"type": "intermediary_result", "stage": "sql_fixer", "name": "Исправленный SQL", "content": validated_sql, "language": "sql"}
        
        accumulated_data_for_rag["sql_query"] = validated_sql

        # 5. Выполнение SQL
        current_stage = "Выполнение SQL"
        yield {"type": "status", "stage": "sql_execution", "message": current_stage + "..."}
        try:
            sql_results_df = await execute_duckdb_query(duckdb_con, validated_sql)
            if sql_results_df is not None and not sql_results_df.empty:
                # Преобразуем datetime в строки
                for col in sql_results_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sql_results_df[col]):
                        sql_results_df[col] = sql_results_df[col].astype(str)
                
                # ОГРАНИЧИВАЕМ ВОЗВРАЩАЕМЫЕ ДАННЫЕ ДО 100 СТРОК
                display_df = sql_results_df.head(100)
                sql_results_json = display_df.to_dict(orient='records')
                
                yield {"type": "sql_result", "stage": "sql_execution", "data": sql_results_json, "row_count": len(sql_results_df), "message": f"SQL выполнен. Показано {len(display_df)} из {len(sql_results_df)} строк."}
            else:
                sql_results_json = []
                yield {"type": "sql_result", "stage": "sql_execution", "data": [], "row_count": 0, "message": "SQL выполнен, но данных нет."}
        except Exception as e:
            yield {"type": "error", "stage": "sql_execution", "message": f"Ошибка SQL: {str(e)}"}
            return

        # 6. ОПТИМИЗИРОВАННЫЙ анализ аномалий (топ-10)
        anomaly_summary = None
        if needs_anomaly_detection:
            current_stage = "Анализ аномалий (топ-10)"
            yield {"type": "status", "stage": "anomaly_detection", "message": current_stage + "..."}
            if sql_results_df is not None and not sql_results_df.empty:
                # Ограничиваем анализ первыми 1000 строками для производительности
                analysis_df = sql_results_df.head(1000)
                anomaly_results = agent_anomaly_detector(analysis_df, processed_user_prompt)
                yield {"type": "intermediary_result", "stage": "anomaly_detection", "name": "Аномалии (топ-10)", "content": anomaly_results}
                anomaly_summary = anomaly_results
            else:
                yield {"type": "warning", "stage": "anomaly_detection", "message": "Нет данных для анализа аномалий."}

        # 7. КОМПАКТНАЯ генерация итогового ответа
        current_stage = "Генерация компактного ответа"
        yield {"type": "status", "stage": "final_summary_generation", "message": current_stage + "..."}
        llm_summary_result = generate_llm_analysis_summary(
            sql_results_df, 
            processed_user_prompt, 
            anomaly_results=anomaly_summary,
            needs_anomaly_detection=needs_anomaly_detection,
            rag_context=rag_context_for_agents
        )
        accumulated_data_for_rag["final_answer"] = llm_summary_result
        yield {"type": "final_answer", "stage": "final_summary_generation", "answer": llm_summary_result, "rag_components": accumulated_data_for_rag, "message": "Компактный ответ готов."}

    except Exception as e:
        error_message = f"Критическая ошибка на этапе '{current_stage}': {str(e)}"
        print(f"[EVENT_GENERATOR ERROR] {error_message}")
        yield {"type": "error", "stage": current_stage, "message": error_message}
    finally:
        yield {"type": "status", "stage": "done", "message": "Обработка завершена."}

# ==============================================================================
# ЭНДПОИНТЫ (БЕЗ ИЗМЕНЕНИЙ)
# ==============================================================================

@app.get("/process_query_stream")
async def process_query_stream(user_prompt: str, request: Request):
    """Endpoint для обработки запросов с потоковым ответом"""
    duckdb_con = request.app.state.duckdb_con
    if not duckdb_con:
        return JSONResponse(status_code=500, content={"error": "База данных не инициализирована."})
    
    if not user_prompt or not user_prompt.strip():
        return JSONResponse(status_code=400, content={"error": "user_prompt не предоставлен или пуст."})

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
            return {"status": "healthy", "service": "text2sql-optimized", "database": "connected"}
        else:
            return {"status": "unhealthy", "service": "text2sql-optimized", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "service": "text2sql-optimized", "error": str(e)}

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
# EXCEPTION HANDLERS И STARTUP/SHUTDOWN
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
    print("🚀 Запуск Optimized Text2SQL Multi-Agent System...")
    
    try:
        duckdb_connection = setup_duckdb()
        if duckdb_connection is not None:
            app.state.duckdb_con = duckdb_connection
            print("✅ База данных DuckDB инициализирована")
        else:
            print("❌ Не удалось инициализировать базу данных")
            app.state.duckdb_con = None

        # Initialize RAG database
        print("Инициализация RAG базы знаний...")
        if not init_rag_db():
            print("[STARTUP ERROR] Не удалось инициализировать RAG базу.")
        else:
            print("✅ RAG база знаний инициализирована")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        app.state.duckdb_con = None

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при завершении"""
    print("🛑 Завершение работы Optimized Text2SQL System...")
    
    try:
        if (hasattr(app.state, 'duckdb_con') and 
            app.state.duckdb_con is not None and
            hasattr(app.state.duckdb_con, 'close')):
            app.state.duckdb_con.close()
            print("✅ Соединение с базой данных закрыто")
    except Exception as e:
        print(f"⚠️ Ошибка при закрытии соединения: {e}")

# ==============================================================================
# RAG SAVE ENDPOINT
# ==============================================================================

class RagSaveRequest(BaseModel):
    user_prompt: str
    formal_prompt: str
    plan: str
    sql_query: str
    final_answer: str

@app.post("/llm_query/confirm_and_save")
async def confirm_answer_and_save_to_rag(payload: RagSaveRequest):
    """Endpoint для сохранения подтвержденного ответа в RAG базу"""
    print(f"[API /confirm_and_save] Сохранение в RAG: {payload.user_prompt[:50]}...")
    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, lambda: add_to_rag_db(
            user_prompt=payload.user_prompt,
            formal_prompt=payload.formal_prompt,
            plan=payload.plan,
            sql_query=payload.sql_query,
            final_answer=payload.final_answer
        ))

        if success:
            print(f"[API /confirm_and_save] Успешно сохранено: {payload.user_prompt[:50]}")
            return JSONResponse(content={"status": "success", "message": "Ответ сохранен в базу знаний."}, status_code=200)
        else:
            print(f"[API /confirm_and_save] Ошибка сохранения: {payload.user_prompt[:50]}")
            raise HTTPException(status_code=500, detail="Не удалось сохранить в базу знаний.")
    except Exception as e:
        print(f"[API /confirm_and_save] Исключение при сохранении: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения в RAG: {str(e)}")

# ==============================================================================
# ЗАПУСК СЕРВЕРА
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Запуск Optimized Text2SQL Multi-Agent System...")
    
    if not os.path.isdir(STATIC_DIR):
        print(f"[CRITICAL ERROR] Static директория не найдена: {STATIC_DIR}")
    elif not os.path.exists(INDEX_HTML_PATH):
        print(f"[CRITICAL ERROR] index.html не найден: {INDEX_HTML_PATH}")
    
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5002,
        reload=False,
        log_level="info",
        access_log=True
    )