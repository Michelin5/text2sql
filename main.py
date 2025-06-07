import asyncio
import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
from json.decoder import JSONDecodeError
from flask import Flask, request, jsonify
from vis_sber import visualize_report

# Импортируем агентов и утилиты
from giga_wrapper import giga_client, call_giga_api_wrapper
from agents import (
    agent0_coordinator,
    agent1_rephraser,
    agent2_planner,
    agent3_sql_generator,
    agent_sql_validator,
    agent_sql_fixer,
    agent_anomaly_detector
)
from duckdb_utils import setup_duckdb, execute_duckdb_query

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def generate_llm_analysis_summary(sql_results_df, user_prompt, anomaly_results=None, needs_anomaly_detection=False):
    """
    Генерирует LLM-анализ результатов с структурированным саммари
    """
    try:
        # Подготавливаем данные для анализа
        analysis_context = prepare_analysis_context(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)

        # Генерируем LLM-анализ
        llm_summary = generate_intelligent_summary(analysis_context)

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

def generate_intelligent_summary(analysis_context):
    """Генерирует интеллектуальное саммари с помощью LLM"""
    system_instruction = """
Ты — эксперт-аналитик данных. На основе предоставленного контекста анализа создай структурированное, профессиональное саммари на русском языке.

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
    Асинхронный генератор с LLM-анализом результатов
    """
    anomaly_results = None  # Сохраняем результаты анализа аномалий

    try:
        print(f"\n[Stream] Получен запрос: {user_prompt}")

        # --- Анализ неопределенности (встроенный) ---
        ambiguity_data, enhanced_prompt, needs_clarification, needs_anomaly_detection = await analyze_query_ambiguity(user_prompt)

        if needs_clarification:
            yield f"data: {json.dumps({'step': 'clarification_needed', 'content': ambiguity_data}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'step': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
            return

        if ambiguity_data and enhanced_prompt != user_prompt:
            yield f"data: {json.dumps({'step': 'query_enhanced', 'content': ambiguity_data}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
            user_prompt = enhanced_prompt
            print(f"[Stream] Запрос автоматически улучшен: {user_prompt}")

        # --- Агент 0: Координатор ---
        try:
            coordination_result = agent0_coordinator(user_prompt)

            if isinstance(coordination_result, str):
                print(f"[COORDINATOR WARNING] Получена строка: {coordination_result}")
                try:
                    import re
                    json_match = re.search(r'\{.*\}', coordination_result, re.DOTALL)
                    if json_match:
                        coordination_result = json.loads(json_match.group(0))
                    else:
                        raise ValueError("JSON не найден в ответе")
                except Exception as parse_error:
                    print(f"[COORDINATOR ERROR] Ошибка парсинга: {parse_error}")
                    coordination_result = {
                        "needs_anomaly_detection": needs_anomaly_detection,
                        "analysis_type": "anomaly" if needs_anomaly_detection else "standard",
                        "keywords": []
                    }

            if coordination_result.get("needs_anomaly_detection") is not None:
                needs_anomaly_detection = coordination_result.get("needs_anomaly_detection", False)

            print(f"[Stream] Результат координации: {coordination_result}")
            yield f"data: {json.dumps({'step': 'coordination', 'content': coordination_result}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        except Exception as coord_error:
            print(f"[COORDINATOR ERROR] Ошибка в координаторе: {coord_error}")
            coordination_result = {
                "needs_anomaly_detection": needs_anomaly_detection,
                "analysis_type": "anomaly" if needs_anomaly_detection else "standard",
                "keywords": [],
                "error": str(coord_error)
            }
            yield f"data: {json.dumps({'step': 'coordination', 'content': coordination_result}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # --- Агент 1: Формализация запроса ---
        try:
            formal_request = agent1_rephraser(user_prompt)
            print(f"[Stream] Формальный запрос: {formal_request}")
            yield f"data: {json.dumps({'step': 'formal_request', 'content': formal_request}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[REPHRASER ERROR] Ошибка в переформулировании: {e}")
            formal_request = user_prompt
            yield f"data: {json.dumps({'step': 'formal_request', 'content': formal_request}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # --- Агент 2: Построение плана ---
        try:
            plan = agent2_planner(formal_request)
            print(f"[Stream] План: {plan}")
            yield f"data: {json.dumps({'step': 'plan', 'content': plan}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[PLANNER ERROR] Ошибка в планировании: {e}")
            plan = f"1. Анализ запроса: {formal_request}\n2. Поиск релевантных данных\n3. Формирование ответа"
            yield f"data: {json.dumps({'step': 'plan', 'content': plan}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # --- Агент 3: Генерация SQL ---
        try:
            generated_sql_query = agent3_sql_generator(plan)
            print(f"[Stream] Сгенерированный SQL: {generated_sql_query}")
            yield f"data: {json.dumps({'step': 'generated_sql_query', 'content': generated_sql_query}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[SQL_GENERATOR ERROR] Ошибка в генерации SQL: {e}")
            yield f"data: {json.dumps({'step': 'error', 'content': f'Ошибка генерации SQL: {str(e)}'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'step': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
            return

        current_sql_query = generated_sql_query
        sql_results_df = None
        MAX_FIX_ATTEMPTS = 2

        # --- Цикл валидации и исправления SQL ---
        for attempt in range(MAX_FIX_ATTEMPTS):
            if not current_sql_query or "SELECT" not in current_sql_query.upper():
                error_msg = 'Не удалось сгенерировать корректный SQL-запрос. Проверьте формулировку запроса.'
                yield f"data: {json.dumps({'step': 'final_answer', 'content': error_msg}, ensure_ascii=False)}\n\n"
                break

            yield f"data: {json.dumps({'step': 'executed_sql_query', 'content': current_sql_query}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

            try:
                print(f"[Stream] Попытка выполнения SQL ({attempt + 1}/{MAX_FIX_ATTEMPTS})...")
                sql_results_df = execute_duckdb_query(current_sql_query)

                if sql_results_df is not None and not sql_results_df.empty:
                    sql_results_str = sql_results_df.to_markdown(index=False)
                    print(f"[Stream] SQL выполнен успешно. Получено {len(sql_results_df)} строк.")
                else:
                    sql_results_str = "Запрос SQL выполнен, но данные не найдены."
                    print("[Stream] SQL выполнен, но результат пустой.")

                yield f"data: {json.dumps({'step': 'sql_results_str', 'content': sql_results_str}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
                break



            except Exception as e:
                sql_error_message = str(e)
                print(f"[Stream] Ошибка DuckDB: {sql_error_message}")
                yield f"data: {json.dumps({'step': 'sql_error', 'content': sql_error_message}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

                if attempt < MAX_FIX_ATTEMPTS - 1:
                    print("[Stream] Попытка исправить SQL...")
                    try:
                        fixed_sql = agent_sql_fixer(current_sql_query, sql_error_message, plan)
                        fix_log_entry = {
                            "fix_attempt": attempt + 1,
                            "fixed_sql": fixed_sql,
                            "original_sql": current_sql_query,
                            "error": sql_error_message
                        }
                        yield f"data: {json.dumps({'step': 'sql_validation_log', 'content': fix_log_entry}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)

                        if fixed_sql and fixed_sql.strip():
                            print(f"[Stream] Fixer предложил новый SQL: {fixed_sql}")
                            current_sql_query = fixed_sql
                        else:
                            print("[Stream] Fixer не смог предложить исправление.")
                            sql_results_df = None
                            break
                    except Exception as fixer_error:
                        print(f"[FIXER ERROR] Ошибка в исправлении SQL: {fixer_error}")
                        sql_results_df = None
                        break
                else:
                    print("[Stream] Достигнуто макс. число попыток исправления.")
                    sql_results_df = None

        # --- Агент аномалий (если требуется) ---
        if needs_anomaly_detection:
            print("[Stream] Запуск агента обнаружения аномалий...")
            try:
                anomaly_results = agent_anomaly_detector(sql_results_df, user_prompt)
                print(f"[Stream] Результаты анализа аномалий: {anomaly_results}")
                yield f"data: {json.dumps({'step': 'anomaly_analysis', 'content': anomaly_results}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[ANOMALY ERROR] Ошибка в агенте аномалий: {e}")
                anomaly_results = {
                    "anomalies_found": False,
                    "error": f"Ошибка анализа аномалий: {str(e)}",
                    "message": "Не удалось выполнить анализ аномалий"
                }
                yield f"data: {json.dumps({'step': 'anomaly_analysis', 'content': anomaly_results}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

        # --- LLM-АНАЛИЗ И ГЕНЕРАЦИЯ САММАРИ ---
        print("[Stream] Генерация LLM-саммари результатов...")
        yield f"data: {json.dumps({'step': 'llm_analysis_start', 'content': 'Генерация интеллектуального саммари...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)

        try:
            llm_summary = generate_llm_analysis_summary(
                sql_results_df,
                user_prompt,
                anomaly_results,
                needs_anomaly_detection
            )
            print("[Stream] LLM-саммари сгенерировано.")
            yield f"data: {json.dumps({'step': 'final_answer', 'content': llm_summary}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[LLM_SUMMARY ERROR] Ошибка в генерации LLM-саммари: {e}")
            fallback_summary = generate_fallback_summary(sql_results_df, user_prompt, anomaly_results, needs_anomaly_detection)
            yield f"data: {json.dumps({'step': 'final_answer', 'content': fallback_summary}, ensure_ascii=False)}\n\n"

    except Exception as e:
        print(f"[Stream] Произошла критическая ошибка в генераторе: {e}")
        error_content = f"**❌ Критическая ошибка системы**\n\nПроизошла неожиданная ошибка: {e}\n\nПожалуйста, попробуйте еще раз или обратитесь к администратору."
        yield f"data: {json.dumps({'step': 'error', 'content': error_content}, ensure_ascii=False)}\n\n"

    finally:
        print("[Stream] Отправка сигнала о завершении.")
        yield f"data: {json.dumps({'step': 'done', 'content': ''}, ensure_ascii=False)}\n\n"

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

    return StreamingResponse(
        event_generator(user_prompt.strip(), duckdb_con),
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

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Text2SQL Multi-Agent System with LLM-Enhanced Analysis",
        "version": "2.3.0",
        "features": [
            "Intelligent query coordination",
            "Advanced anomaly detection with LLM analysis",
            "Automated ambiguity resolution",
            "Multi-agent processing pipeline",
            "Real-time streaming responses",
            "LLM-generated structured summaries"
        ],
        "supported_queries": [
            "Standard data analysis with intelligent summaries",
            "Anomaly detection with expert-level insights",
            "Statistical analysis with actionable recommendations",
            "Data exploration with comprehensive reporting"
        ]
    }

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

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['result'])
        status, content = visualize_report(df)
        return jsonify({'status': status, 'content': content})
    except Exception as e:
        return jsonify({'status': 'error', 'content': str(e)})

# ==============================================================================
# ЗАПУСК СЕРВЕРА
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Запуск Text2SQL Multi-Agent System with LLM Analysis...")

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        reload=False,
        log_level="info",
        access_log=True
    )
