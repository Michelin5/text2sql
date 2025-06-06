import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
from json.decoder import JSONDecodeError

# Импортируем ваших агентов и утилиты
from giga_wrapper import giga_client, call_giga_api_wrapper # Убедитесь, что эта обертка надежна
from agents import (
    agent1_rephraser,
    agent2_planner,
    agent3_sql_generator,
    agent_sql_validator,
    agent_sql_fixer,
    agent4_answer_generator
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

# Константы и TABLE_CONTEXT остаются без изменений
TABLE_CONTEXT = """
... ваш контекст таблиц ...
"""

# ==============================================================================
# БЛОК 1: ФУНКЦИЯ-ГЕНЕРАТОР (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ==============================================================================
async def event_generator(user_prompt: str, duckdb_con):
    """
    Асинхронный генератор, который выполняет весь конвейер агентов
    и НАПРЯМУЮ отдает (yield) результаты каждого шага.
    """
    try:
        print(f"\n[Stream] Получен запрос: {user_prompt}")

        # --- Агент 1: Формализация запроса ---
        formal_request = agent1_rephraser(user_prompt)
        print(f"[Stream] Формальный запрос: {formal_request}")
        yield f"data: {json.dumps({'step': 'formal_request', 'content': formal_request})}\n\n"
        await asyncio.sleep(0.01)

        # --- Агент 2: Построение плана ---
        plan = agent2_planner(formal_request)
        print(f"[Stream] План: {plan}")
        yield f"data: {json.dumps({'step': 'plan', 'content': plan})}\n\n"
        await asyncio.sleep(0.01)
        
        # --- Агент 3: Генерация SQL ---
        generated_sql_query = agent3_sql_generator(plan)
        print(f"[Stream] Сгенерированный SQL: {generated_sql_query}")
        yield f"data: {json.dumps({'step': 'generated_sql_query', 'content': generated_sql_query})}\n\n"
        await asyncio.sleep(0.01)

        current_sql_query = generated_sql_query
        sql_results_df = None
        MAX_FIX_ATTEMPTS = 2

        # --- Цикл валидации и исправления SQL ---
        for attempt in range(MAX_FIX_ATTEMPTS):
            if not current_sql_query or "SELECT" not in current_sql_query.upper():
                yield f"data: {json.dumps({'step': 'final_answer', 'content': 'Не удалось сгенерировать корректный SQL-запрос.'})}\n\n"
                break

            yield f"data: {json.dumps({'step': 'executed_sql_query', 'content': current_sql_query})}\n\n"
            await asyncio.sleep(0.01)
            
            try:
                print(f"[Stream] Попытка выполнения SQL ({attempt + 1}/{MAX_FIX_ATTEMPTS})...")
                sql_results_df = execute_duckdb_query(current_sql_query)
                sql_results_str = sql_results_df.to_markdown(index=False) if not sql_results_df.empty else "Запрос SQL выполнен, но данные не найдены."
                
                print("[Stream] SQL выполнен успешно.")
                yield f"data: {json.dumps({'step': 'sql_results_str', 'content': sql_results_str})}\n\n"
                await asyncio.sleep(0.01)
                break  # Выход из цикла при успехе

            except Exception as e:
                sql_error_message = str(e)
                print(f"[Stream] Ошибка DuckDB: {sql_error_message}")
                yield f"data: {json.dumps({'step': 'sql_error', 'content': sql_error_message})}\n\n"
                await asyncio.sleep(0.01)

                if attempt < MAX_FIX_ATTEMPTS - 1:
                    print("[Stream] Попытка исправить SQL...")
                    fixed_sql = agent_sql_fixer(current_sql_query, sql_error_message, plan)
                    fix_log_entry = {"fix_attempt": fixed_sql, "original_sql": current_sql_query, "error": sql_error_message}
                    yield f"data: {json.dumps({'step': 'sql_validation_log', 'content': fix_log_entry})}\n\n"
                    await asyncio.sleep(0.01)

                    if fixed_sql:
                        print(f"[Stream] Fixer предложил новый SQL: {fixed_sql}")
                        current_sql_query = fixed_sql
                    else:
                        print("[Stream] Fixer не смог предложить исправление. Прерывание.")
                        sql_results_df = None # Убедимся, что df не используется дальше
                        break
                else:
                    print("[Stream] Достигнуто макс. число попыток исправления.")
                    sql_results_df = None # Убедимся, что df не используется дальше

        # --- Финальный ответ ---
        if sql_results_df is not None:
             final_answer = agent4_answer_generator(sql_results_df, user_prompt)
             print("[Stream] Финальный ответ сгенерирован.")
             yield f"data: {json.dumps({'step': 'final_answer', 'content': final_answer})}\n\n"
        else:
             final_answer = f"Не удалось выполнить SQL-запрос после {MAX_FIX_ATTEMPTS} попыток. Проверьте ваш запрос или обратитесь к администратору."
             yield f"data: {json.dumps({'step': 'final_answer', 'content': final_answer})}\n\n"

    except Exception as e:
        print(f"[Stream] Произошла критическая ошибка в генераторе: {e}")
        error_content = f"Произошла критическая ошибка на сервере: {e}"
        yield f"data: {json.dumps({'step': 'error', 'content': error_content})}\n\n"

    finally:
        # --- Сигнал завершения потока ---
        print("[Stream] Отправка сигнала о завершении.")
        yield f"data: {json.dumps({'step': 'done', 'content': ''})}\n\n"


# ==============================================================================
# БЛОК 2: ФУНКЦИЯ-ОБРАБОТЧИК ЭНДПОИНТА (короткая и чистая)
# ==============================================================================
@app.get("/process_query_stream")
async def process_query_stream(user_prompt: str, request: Request):
    """
    Endpoint, который принимает user_prompt через GET-запрос 
    и возвращает потоковый ответ, запуская event_generator.
    """
    duckdb_con = request.app.state.duckdb_con
    if not duckdb_con:
        return JSONResponse(status_code=500, content={"error": "База данных не инициализирована."})
    
    if not user_prompt:
        return JSONResponse(status_code=400, content={"error": "user_prompt не предоставлен."})

    # Эта строка ЗАПУСКАЕТ всю логику из Блока 1
    return StreamingResponse(event_generator(user_prompt, duckdb_con), media_type="text/event-stream")


# ==============================================================================
# ЗАПУСК СЕРВЕРА
# ==============================================================================
if __name__ == "__main__":
    # Убедитесь, что эта функция корректно настраивает все необходимое
    app.state.duckdb_con = setup_duckdb()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)