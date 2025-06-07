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
    agent3_sql_generator_advanced,  # Используем продвинутый генератор
    agent_sql_validator,
    agent_sql_fixer,
    agent_anomaly_detector,
    agent_rag_retriever,
    calculate_query_similarity,  # Для 90% совпадений
    generate_fallback_sql,       # Fallback функции
    extract_sql_from_response,
    validate_and_explain_sql
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
# ФУНКЦИИ ПОДДЕРЖКИ 90% СОВПАДЕНИЙ
# ==============================================================================

def check_high_similarity_with_rag(user_prompt: str, rag_data: dict) -> bool:
    """Проверяет, превышает ли сходство с прошлым запросом 90%"""
    if not rag_data or not isinstance(rag_data, dict):
        return False
    
    past_prompt = rag_data.get('user_prompt', '')
    if not past_prompt:
        return False
    
    similarity_score = calculate_query_similarity(user_prompt, past_prompt)
    
    print(f"[SIMILARITY_CHECK] Текущий: '{user_prompt[:50]}...'")
    print(f"[SIMILARITY_CHECK] Прошлый: '{past_prompt[:50]}...'")
    print(f"[SIMILARITY_CHECK] Сходство: {similarity_score:.3f}")
    
    return similarity_score >= 0.90

# ==============================================================================
# АНАЛИЗ НЕОПРЕДЕЛЕННОСТИ С ИНТЕГРАЦИЕЙ AGENTS.PY
# ==============================================================================

async def analyze_query_ambiguity(user_prompt):
    """Анализирует неопределенность запроса с интеграцией coordinator"""
    prompt_lower = user_prompt.lower()
    
    # Используем координатор из agents.py для определения типа анализа
    try:
        coordination_result = agent0_coordinator(user_prompt)
        
        # Обрабатываем результат координатора
        if isinstance(coordination_result, str):
            import re
            json_match = re.search(r'\{.*\}', coordination_result, re.DOTALL)
            if json_match:
                coordination_result = json.loads(json_match.group(0))
            else:
                coordination_result = {
                    "needs_anomaly_detection": False,
                    "analysis_type": "standard",
                    "keywords": []
                }
        
        needs_anomaly_detection = coordination_result.get('needs_anomaly_detection', False)
        analysis_type = coordination_result.get('analysis_type', 'standard')
        
        print(f"[COORDINATOR] Результат: {coordination_result}")
        
    except Exception as e:
        print(f"[COORDINATOR ERROR] {e}")
        # Fallback анализ
        anomaly_keywords = ['аномали', 'выброс', 'необычн', 'странн', 'отклонени']
        needs_anomaly_detection = any(keyword in prompt_lower for keyword in anomaly_keywords)
        analysis_type = "anomaly" if needs_anomaly_detection else "standard"
    
    # Детекция таблиц
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
    
    # Автоматическое улучшение запросов
    if needs_anomaly_detection and not mentioned_tables:
        enhanced_prompt = f"Найди аномалии в данных о населении"
        enhancement_data = {
            "ambiguity_level": "medium",
            "original_prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt,
            "reason": "Автоматически выбрана таблица 'население' для поиска аномалий",
            "auto_selected_table": "population"
        }
        return enhancement_data, enhanced_prompt, False, needs_anomaly_detection
    
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
            return enhancement_data, enhanced_prompt, False, needs_anomaly_detection
    
    # Требуется уточнение только для очень коротких запросов
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
        return clarification_data, None, True, needs_anomaly_detection
    
    return None, user_prompt, False, needs_anomaly_detection

# ==============================================================================
# ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА С КОМПАКТНОСТЬЮ
# ==============================================================================

def generate_enhanced_final_response(sql_results_df, user_prompt, anomaly_results=None, needs_anomaly_detection=False, analysis_steps=0, analysis_context=None):
    """Генерирует компактный финальный ответ с внутренней цепью рассуждений"""
    
    anomalies_found = (anomaly_results is not None and anomaly_results.get('anomalies_found', False))
    
    if needs_anomaly_detection:
        status_icon = "🚨" if anomalies_found else "✅"
        status_text = "Аномалии обнаружены" if anomalies_found else "Аномалии не найдены"
    else:
        status_icon = "📊"
        status_text = "Анализ завершен"
    
    # ВНУТРЕННЯЯ ЦЕПЬ РАССУЖДЕНИЙ (НЕ ПОКАЗЫВАЕМ ПОЛЬЗОВАТЕЛЮ)
    internal_reasoning = f"""
    Шаг 1: Анализ данных - получено {len(sql_results_df) if sql_results_df is not None else 0} записей
    Шаг 2: Тип анализа - {'поиск аномалий' if needs_anomaly_detection else 'стандартный анализ'}
    Шаг 3: Результат аномалий - {'найдены' if anomalies_found else 'не найдены'}
    Шаг 4: Качество данных - {'требует внимания' if anomalies_found else 'в норме'}
    Шаг 5: Chain of thought шагов - {analysis_steps}
    """
    
    # Генерируем краткий LLM ответ
    try:
        system_instruction = f"""
        Создай краткий (максимум 80 слов) ответ на русском языке.
        
        Внутренняя цепь рассуждений (НЕ показывай её пользователю):
        {internal_reasoning}
        
        Структура ответа:
        1. Краткое резюме (1 предложение)
        2. Ключевая информация о данных
        3. Практический вывод
        
        Используй дружелюбный тон.
        """
        
        prompt = f"""
        Запрос: "{user_prompt}"
        Результат: {status_text}
        Записей: {len(sql_results_df) if sql_results_df is not None else 0}
        """
        
        if anomaly_results and anomalies_found:
            prompt += f"Найдено аномалий: {anomaly_results.get('anomaly_count', 'несколько')}\n"
        
        llm_text = call_giga_api_wrapper(prompt, system_instruction)
    except Exception as e:
        llm_text = f"Анализ завершен. Обработано {len(sql_results_df) if sql_results_df is not None else 0:,} записей."
    
    response = f"{status_icon} **{status_text}**\n\n{llm_text}\n\n"
    
    # ПЕРВЫЕ 10-15 СТРОК ДАННЫХ
    if sql_results_df is not None and not sql_results_df.empty:
        response += "**📋 Образец данных (первые 15 строк):**\n"
        display_df = sql_results_df.head(15)
        response += display_df.to_markdown(index=False) + "\n"
        if len(sql_results_df) > 15:
            response += f"*Показаны первые 15 из {len(sql_results_df):,} записей*\n\n"
    
    # ХАРАКТЕРИСТИКА ДАННЫХ
    if sql_results_df is not None and not sql_results_df.empty:
        numeric_cols = sql_results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            key_col = numeric_cols[0]
            col_data = sql_results_df[key_col].dropna()
            response += f"**📈 Характеристика данных:**\n"
            response += f"• Основной показатель: {key_col}\n"
            response += f"• Среднее значение: {col_data.mean():.1f}\n"
            response += f"• Диапазон: {col_data.min():.1f} — {col_data.max():.1f}\n"
            response += f"• Стандартное отклонение: {col_data.std():.1f}\n\n"
    
    # КОНКРЕТНЫЕ ПРИМЕРЫ АНОМАЛИЙ (ТОЛЬКО ТОП-10)
    if anomalies_found:
        response += "**🎯 Найденные аномалии (топ-10):**\n"
        
        if 'anomalies' in anomaly_results:
            for i, anomaly in enumerate(anomaly_results['anomalies'][:10], 1):
                if isinstance(anomaly, dict):
                    territory_name = anomaly.get('territory', anomaly.get('column', f'Объект {i}'))
                    count = anomaly.get('anomaly_count', 0)
                    examples = anomaly.get('outlier_values', [])
                    
                    response += f"{i}. **{territory_name}**: {count} аномальных значений\n"
                    if examples:
                        examples_str = ", ".join([f"{x:.1f}" for x in examples[:3]])
                        response += f"   Примеры: {examples_str}\n"
        response += "\n"
    else:
        response += "**✅ Аномалии не найдены** - все значения в пределах нормы\n\n"
    
    # КРАТКАЯ ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ
    response += f"**📊 Итого:** {len(sql_results_df) if sql_results_df is not None else 0:,} записей"
    if anomalies_found and 'anomalies' in anomaly_results:
        total_anomalies = sum(a.get('anomaly_count', 0) for a in anomaly_results['anomalies'][:10] if isinstance(a, dict))
        response += f" • {total_anomalies} аномалий"
    response += f" • {analysis_steps} этапов анализа"
    
    return response

# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ-ГЕНЕРАТОР С ИНТЕГРАЦИЕЙ AGENTS.PY
# ==============================================================================

async def event_generator(user_prompt: str, duckdb_con):
    """
    Полностью интегрированный генератор событий с поддержкой всех функций из agents.py
    """
    current_stage = "Начало обработки"
    analysis_context = {}
    rag_context_for_agents = None
    cot_steps = []
    
    # Accumulated data for potential saving
    accumulated_data_for_rag = {
        "user_prompt": user_prompt,
        "formal_prompt": None,
        "plan": None,
        "sql_query": None,
        "final_answer": None
    }

    async def add_cot_step(step_text, step_type="thinking", is_complete=False):
        """Добавляет шаг в цепочку рассуждений"""
        cot_steps.append({
            "text": step_text,
            "type": step_type,
            "complete": is_complete,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        yield {"type": "cot_step", "stage": "chain_of_thought", 
               "content": {"steps": cot_steps, "current_step": step_text, "complete": is_complete}}
        await asyncio.sleep(0.3)

    try:
        # Начинаем цепочку рассуждений
        yield {"type": "cot_start", "stage": "reasoning", "message": "Начинаю интеллектуальный анализ..."}
        await asyncio.sleep(0.3)

        # === ЭТАП 1: RAG С ПОДДЕРЖКОЙ 90% СОВПАДЕНИЙ ===
        async for cot_event in add_cot_step("🔍 Проверяю базу знаний на наличие похожих запросов"):
            yield cot_event

        rag_result = agent_rag_retriever(user_prompt)
        rag_status = rag_result.get("status")
        rag_data = rag_result.get("data")

        if rag_status == "error":
            async for cot_event in add_cot_step("⚠️ Ошибка базы знаний, продолжаю без неё", "warning", True):
                yield cot_event
            yield {"type": "warning", "stage": "rag_check", "message": f"Ошибка базы знаний: {rag_result.get('message')}"}
            
        elif rag_status == "exact":
            async for cot_event in add_cot_step("✅ Найден точный ответ в базе знаний", "success", True):
                yield cot_event
            
            final_answer = rag_data.get("final_answer", "Точный ответ найден.")
            accumulated_data_for_rag.update(rag_data)
            
            yield {"type": "cot_hide", "stage": "reasoning", "message": ""}
            await asyncio.sleep(0.5)
            yield {"type": "final_answer", "stage": "rag_exact_match", "answer": final_answer, 
                   "rag_components": accumulated_data_for_rag, "message": "Найден точный ответ."}
            return
            
        elif rag_status == "similar":
            # ПРОВЕРКА 90% СОВПАДЕНИЯ
            if check_high_similarity_with_rag(user_prompt, rag_data):
                async for cot_event in add_cot_step("🎯 Обнаружено 90%+ совпадение с прошлым запросом", "success", True):
                    yield cot_event
                
                final_answer = rag_data.get("final_answer", "Найден очень похожий ответ.")
                accumulated_data_for_rag.update(rag_data)
                
                yield {"type": "cot_hide", "stage": "reasoning", "message": ""}
                await asyncio.sleep(0.5)
                yield {"type": "final_answer", "stage": "rag_high_similarity_match", "answer": final_answer,
                       "rag_components": accumulated_data_for_rag, "message": "Найден ответ на очень похожий запрос (совпадение >90%)."}
                return
            else:
                async for cot_event in add_cot_step("📋 Найден похожий запрос, использую как контекст", "info", True):
                    yield cot_event
                rag_context_for_agents = rag_data
        else:
            async for cot_event in add_cot_step("🆕 Новый запрос, стандартная обработка", "info", True):
                yield cot_event

        # === ЭТАП 2: АНАЛИЗ НЕОПРЕДЕЛЕННОСТИ ===
        async for cot_event in add_cot_step("🔍 Анализирую тип запроса и определяю стратегию"):
            yield cot_event

        current_stage = "Анализ неопределенности"
        ambiguity_analysis_result, enhanced_prompt, needs_clarification, needs_anomaly_detection = await analyze_query_ambiguity(user_prompt)
        
        if needs_clarification:
            async for cot_event in add_cot_step("❓ Запрос требует уточнения", "warning", True):
                yield cot_event
            
            yield {"type": "cot_hide", "stage": "reasoning", "message": ""}
            await asyncio.sleep(0.5)
            yield {"type": "clarification_needed", "stage": "ambiguity_analysis", "data": ambiguity_analysis_result}
            return

        if enhanced_prompt and enhanced_prompt != user_prompt:
            async for cot_event in add_cot_step(f"✨ Автоматически улучшил запрос", "enhancement", True):
                yield cot_event
            processed_user_prompt = enhanced_prompt
            analysis_context['enhanced_query'] = True
        else:
            processed_user_prompt = user_prompt

        strategy_text = "поиск аномалий" if needs_anomaly_detection else "стандартный анализ"
        async for cot_event in add_cot_step(f"🎯 Стратегия определена: {strategy_text}", "success", True):
            yield cot_event

        analysis_context['analysis_type'] = 'anomaly_detection' if needs_anomaly_detection else 'standard_analysis'

        # === ЭТАП 3: ПЕРЕФОРМУЛИРОВКА ===
        async for cot_event in add_cot_step("📝 Формализую запрос с учетом семантического анализа"):
            yield cot_event

        current_stage = "Переформулировка запроса"
        formal_prompt_result = agent1_rephraser(processed_user_prompt, rag_context=rag_context_for_agents)
        accumulated_data_for_rag["formal_prompt"] = formal_prompt_result
        analysis_context['formal_request'] = formal_prompt_result

        async for cot_event in add_cot_step("✅ Запрос формализован с сохранением намерений", "success", True):
            yield cot_event

        yield {"type": "intermediary_result", "stage": "rephraser", "name": "Формальный запрос", "content": formal_prompt_result}

        # === ЭТАП 4: ПЛАНИРОВАНИЕ ===
        async for cot_event in add_cot_step("📋 Составляю детальный план с учетом RAG контекста"):
            yield cot_event

        current_stage = "Создание плана"
        plan_result = agent2_planner(formal_prompt_result, rag_context=rag_context_for_agents)
        accumulated_data_for_rag["plan"] = plan_result
        analysis_context['plan'] = plan_result

        plan_steps_count = len([line for line in plan_result.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])])
        async for cot_event in add_cot_step(f"✅ План готов: {plan_steps_count} этапов анализа", "success", True):
            yield cot_event

        yield {"type": "intermediary_result", "stage": "planner", "name": "План выполнения", "content": plan_result}

        # === ЭТАП 5: ПРОДВИНУТАЯ SQL ГЕНЕРАЦИЯ ===
        async for cot_event in add_cot_step("💾 Генерирую оптимизированный SQL с AI-анализом"):
            yield cot_event

        current_stage = "Продвинутая генерация SQL"
        
        try:
            # Используем продвинутый SQL генератор из agents.py
            sql_result = agent3_sql_generator_advanced(plan_result, rag_context_for_agents, processed_user_prompt)
            
            if isinstance(sql_result, dict):
                if sql_result.get("is_valid", True):
                    generated_sql_query = sql_result["sql"]
                    analysis_context['sql_confidence'] = sql_result.get("confidence_score", 85)
                    
                    # Показываем дополнительную информацию о SQL
                    if sql_result.get("explanation"):
                        async for cot_event in add_cot_step(f"🧠 {sql_result['explanation'][:60]}...", "info", True):
                            yield cot_event
                    
                    yield {"type": "sql_analysis", "stage": "sql_generator", 
                           "content": {
                               "explanation": sql_result.get("explanation", ""),
                               "performance_notes": sql_result.get("performance_notes", ""),
                               "confidence": sql_result.get("confidence_score", 0)
                           }}
                else:
                    async for cot_event in add_cot_step("⚠️ SQL требует дополнительной валидации", "warning", True):
                        yield cot_event
                    generated_sql_query = sql_result["sql"]
            else:
                generated_sql_query = sql_result
                
            analysis_context['sql_query'] = generated_sql_query
            
            async for cot_event in add_cot_step("✅ Оптимизированный SQL готов к выполнению", "success", True):
                yield cot_event

        except Exception as e:
            async for cot_event in add_cot_step("❌ Ошибка генерации SQL, использую fallback", "error", True):
                yield cot_event
            
            # Используем fallback из agents.py
            generated_sql_query = generate_fallback_sql({"data_scale": "medium"})
            analysis_context['sql_query'] = generated_sql_query
            print(f"[SQL_GENERATOR ERROR] {e}")

        yield {"type": "intermediary_result", "stage": "sql_generator", "name": "Оптимизированный SQL", 
               "content": generated_sql_query, "language": "sql"}

        # === ЭТАП 6: ВАЛИДАЦИЯ И ИСПРАВЛЕНИЕ ===
        async for cot_event in add_cot_step("🔍 Валидирую SQL на корректность"):
            yield cot_event

        current_stage = "Валидация SQL"
        validator_result = agent_sql_validator(generated_sql_query, plan_result)

        if validator_result.get("is_valid"):
            validated_sql = validator_result.get("corrected_sql") or generated_sql_query
            async for cot_event in add_cot_step("✅ SQL прошёл валидацию", "success", True):
                yield cot_event
        else:
            async for cot_event in add_cot_step("🔧 Исправляю ошибки в SQL"):
                yield cot_event
            
            fixed_sql = agent_sql_fixer(generated_sql_query, validator_result.get("message", ""), plan_result)
            validated_sql = fixed_sql if fixed_sql and fixed_sql.strip() else generated_sql_query
            
            async for cot_event in add_cot_step("✅ SQL исправлен", "success", True):
                yield cot_event

        accumulated_data_for_rag["sql_query"] = validated_sql
        analysis_context['final_sql'] = validated_sql

        # === ЭТАП 7: ВЫПОЛНЕНИЕ SQL ===
        async for cot_event in add_cot_step("⚡ Выполняю SQL-запрос в базе данных"):
            yield cot_event

        current_stage = "Выполнение SQL"
        try:
            sql_results_df = await execute_duckdb_query(duckdb_con, validated_sql)
            
            if sql_results_df is not None and not sql_results_df.empty:
                # Преобразуем datetime в строки
                for col in sql_results_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sql_results_df[col]):
                        sql_results_df[col] = sql_results_df[col].astype(str)
                
                records_count = len(sql_results_df)
                async for cot_event in add_cot_step(f"✅ Получено {records_count:,} записей из базы данных", "success", True):
                    yield cot_event
                
                # ОГРАНИЧИВАЕМ ОТОБРАЖЕНИЕ ДО 100 СТРОК
                display_df = sql_results_df.head(100)
                sql_results_json = display_df.to_dict(orient='records')
                
                analysis_context['data_retrieved'] = True
                analysis_context['records_count'] = records_count
                analysis_context['display_records'] = len(display_df)
                
                yield {"type": "sql_result", "stage": "sql_execution", "data": sql_results_json, 
                       "row_count": records_count, 
                       "message": f"SQL выполнен успешно. Показано {len(display_df)} из {records_count} записей."}
            else:
                async for cot_event in add_cot_step("⚠️ SQL выполнен, но данные не найдены", "warning", True):
                    yield cot_event
                
                sql_results_json = []
                analysis_context['data_retrieved'] = False
                analysis_context['records_count'] = 0
                
                yield {"type": "sql_result", "stage": "sql_execution", "data": [], "row_count": 0, 
                       "message": "SQL выполнен, но данных нет."}
                
        except Exception as e:
            async for cot_event in add_cot_step(f"❌ Ошибка выполнения SQL: {str(e)[:50]}...", "error", True):
                yield cot_event
            yield {"type": "error", "stage": "sql_execution", "message": f"Ошибка SQL: {str(e)}"}
            return

        # === ЭТАП 8: ОПТИМИЗИРОВАННЫЙ АНАЛИЗ АНОМАЛИЙ (ТОП-10) ===
        anomaly_summary = None
        if needs_anomaly_detection:
            async for cot_event in add_cot_step("🔍 Запускаю детектор аномалий (топ-10 результатов)"):
                yield cot_event

            current_stage = "Оптимизированный анализ аномалий"
            if sql_results_df is not None and not sql_results_df.empty:
                # ОГРАНИЧИВАЕМ АНАЛИЗ ПЕРВЫМИ 1000 СТРОКАМИ
                analysis_df = sql_results_df.head(1000)
                anomaly_results = agent_anomaly_detector(analysis_df, processed_user_prompt)
                
                if anomaly_results.get('anomalies_found'):
                    anomaly_count = 0
                    if 'anomalies' in anomaly_results:
                        anomaly_count = sum(a.get('anomaly_count', 0) for a in anomaly_results['anomalies'])
                        async for cot_event in add_cot_step(f"🚨 Обнаружено {anomaly_count} аномальных значений!", "alert", True):
                            yield cot_event
                    else:
                        async for cot_event in add_cot_step("🚨 Найдены статистические аномалии!", "alert", True):
                            yield cot_event
                else:
                    async for cot_event in add_cot_step("✅ Аномалии не найдены - данные в норме", "success", True):
                        yield cot_event

                analysis_context['anomaly_results'] = anomaly_results
                anomaly_summary = anomaly_results
                
                yield {"type": "intermediary_result", "stage": "anomaly_detection", 
                       "name": "Результаты анализа аномалий (топ-10)", "content": anomaly_results}
            else:
                async for cot_event in add_cot_step("⚠️ Нет данных для анализа аномалий", "warning", True):
                    yield cot_event

        # === ЭТАП 9: СКРЫТИЕ ЦЕПОЧКИ РАССУЖДЕНИЙ ===
        async for cot_event in add_cot_step("🧠 Генерирую интеллектуальный финальный анализ"):
            yield cot_event

        await asyncio.sleep(0.5)
        yield {"type": "cot_hide", "stage": "reasoning", "message": ""}
        await asyncio.sleep(0.7)

        # === ЭТАП 10: КОМПАКТНАЯ ГЕНЕРАЦИЯ ИТОГОВОГО ОТВЕТА ===
        current_stage = "Генерация компактного ответа"
        yield {"type": "llm_analysis_start", "stage": "final_summary_generation", 
               "message": "Формирование компактного финального анализа..."}

        try:
            # Генерируем компактный интеллектуальный ответ
            llm_summary_result = generate_enhanced_final_response(
                sql_results_df, 
                processed_user_prompt, 
                anomaly_results=anomaly_summary,
                needs_anomaly_detection=needs_anomaly_detection,
                analysis_steps=len(cot_steps),
                analysis_context=analysis_context
            )
            
            accumulated_data_for_rag["final_answer"] = llm_summary_result
            
            print("[Stream] Компактный интеллектуальный анализ готов.")
            yield {"type": "final_answer", "stage": "final_summary_generation", "answer": llm_summary_result, 
                   "rag_components": accumulated_data_for_rag, "message": "Компактный ответ готов."}
            
        except Exception as e:
            print(f"[FINAL_ANALYSIS ERROR] Ошибка в генерации финального анализа: {e}")
            fallback_summary = f"**📊 Анализ завершен**\n\nОбработано {len(sql_results_df) if sql_results_df is not None else 0} записей по запросу '{processed_user_prompt}'."
            
            accumulated_data_for_rag["final_answer"] = fallback_summary
            yield {"type": "final_answer", "stage": "final_summary_generation", "answer": fallback_summary, 
                   "rag_components": accumulated_data_for_rag, "message": "Резервный ответ готов."}

    except Exception as e:
        error_message = f"Критическая ошибка на этапе '{current_stage}': {str(e)}"
        print(f"[EVENT_GENERATOR ERROR] {error_message}")
        
        # Добавляем ошибку в цепочку рассуждений
        try:
            async for cot_event in add_cot_step(f"❌ Критическая ошибка: {str(e)[:50]}...", "error", True):
                yield cot_event
        except:
            pass
        
        yield {"type": "error", "stage": current_stage, "message": error_message}
    finally:
        print("[Stream] Отправка сигнала о завершении.")
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
            return {"status": "healthy", "service": "text2sql-enhanced", "database": "connected"}
        else:
            return {"status": "unhealthy", "service": "text2sql-enhanced", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "service": "text2sql-enhanced", "error": str(e)}

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
    print("🚀 Запуск Enhanced Text2SQL Multi-Agent System...")
    
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
    print("🛑 Завершение работы Enhanced Text2SQL System...")
    
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
    print("🚀 Запуск Enhanced Text2SQL Multi-Agent System...")
    
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
