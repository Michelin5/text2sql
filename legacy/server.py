# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import duckdb
import pandas as pd
import os
import json

from giga_wrapper import giga_client, call_giga_api_wrapper
from agents import (
    agent1_rephraser,
    agent2_planner,
    agent3_sql_generator,
    agent_sql_validator,
    agent_sql_fixer,
    agent4_answer_generator
)
from duckdb_utils import setup_duckdb, execute_duckdb_query, duckdb_con

# --- Конфигурация Flask ---
app = Flask(__name__)
CORS(app)

# --- Конфигурация автоматического анализа аномалий ---
ANOMALY_AUTO_CONFIG = {
    "intent_keywords": [
        'аномали', 'выброс', 'необычн', 'странн', 'подозрительн', 'отклонен',
        'максимальн', 'минимальн', 'экстремальн', 'качество данных', 'проблем',
        'ошибк', 'зарплат', 'миграци', 'населен', 'сравни', 'динамик', 'тренд',
        'анализ', 'исследован', 'различи', 'распределен', 'выявить', 'найти'
    ],
    "data_criteria": {
        "min_records": 10,
        "min_numeric_columns": 1,
        "min_regions": 3,
        "enable_for_salary": True,
        "enable_for_aggregations": True,
        "enable_for_time_series": True
    },
    "auto_threshold": 2
}

# --- Настройка DuckDB ---
PATH_MARKET_ACCESS = 'data/1_market_access.parquet'
PATH_POPULATION = 'data/2_bdmo_population.parquet'
PATH_MIGRATION = 'data/3_bdmo_migration.parquet'
PATH_SALARY = 'data/4_bdmo_salary.parquet'
PATH_CONNECTIONS = 'data/5_connection.parquet'
PATH_MO_DIRECTORY = 'data/t_dict_municipal_districts.xlsx'

def setup_duckdb():
    global duckdb_con
    mo_directory_created_successfully = False
    all_parquet_views_created = True
    try:
        duckdb_con = duckdb.connect(database=':memory:', read_only=False)
        print("\n--- Инициализация DuckDB и создание представлений (Flask app) ---")
        parquet_files = {
            "market_access": PATH_MARKET_ACCESS, "population": PATH_POPULATION,
            "migration": PATH_MIGRATION, "salary": PATH_SALARY, "connections": PATH_CONNECTIONS
        }
        for view_name, file_path in parquet_files.items():
            if not os.path.exists(file_path):
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Файл Parquet не найден: '{file_path}'. Представление '{view_name}' не будет создано.")
                all_parquet_views_created = False
                continue
            try:
                duckdb_con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{file_path}')")
                print(f"Представление '{view_name}' успешно создано из '{file_path}'.")
            except Exception as e:
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось создать представление '{view_name}' из файла '{file_path}': {e}.")
                all_parquet_views_created = False

        print(f"\nПопытка загрузить справочник МО из файла: {PATH_MO_DIRECTORY}")
        if not os.path.exists(PATH_MO_DIRECTORY):
            print(f"[ОШИБКА] Файл справочника МО не найден по пути: {PATH_MO_DIRECTORY}")
        elif PATH_MO_DIRECTORY.endswith('.xlsx'):
            try:
                print(f"Чтение Excel файла: {PATH_MO_DIRECTORY}...")
                df_mo_directory = pd.read_excel(PATH_MO_DIRECTORY, dtype={'territory_id': str})
                print(f"Excel файл '{PATH_MO_DIRECTORY}' успешно прочитан. Колонки: {df_mo_directory.columns.tolist()}")
                required_cols = ['territory_id', 'municipal_district_name']
                missing_cols = [col for col in required_cols if col not in df_mo_directory.columns]
                if missing_cols:
                    raise ValueError(f"Excel файл справочника МО '{PATH_MO_DIRECTORY}' должен содержать колонки: {', '.join(missing_cols)}.")
                duckdb_con.register('mo_directory_temp_df', df_mo_directory)
                duckdb_con.execute("CREATE VIEW mo_directory AS SELECT CAST(territory_id AS VARCHAR) AS territory_id, municipal_district_name FROM mo_directory_temp_df")
                print(f"Представление 'mo_directory' успешно создано из Excel файла: {PATH_MO_DIRECTORY}")
                mo_directory_created_successfully = True
            except Exception as e:
                print(f"[ОШИБКА] Не удалось обработать Excel файл справочника МО '{PATH_MO_DIRECTORY}': {e}")
        elif PATH_MO_DIRECTORY.endswith('.parquet'):
            try:
                duckdb_con.execute(f"CREATE VIEW mo_directory AS SELECT CAST(territory_id AS VARCHAR) AS territory_id, municipal_district_name FROM read_parquet('{PATH_MO_DIRECTORY}')")
                print(f"Представление 'mo_directory' создано из Parquet файла: {PATH_MO_DIRECTORY}")
                mo_directory_created_successfully = True
            except Exception as e:
                 print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось создать представление 'mo_directory' из Parquet файла '{PATH_MO_DIRECTORY}': {e}")
        elif PATH_MO_DIRECTORY.endswith('.csv'):
            try:
                duckdb_con.execute(f"CREATE VIEW mo_directory AS SELECT CAST(territory_id AS VARCHAR) AS territory_id, \"municipal_district_name\" FROM read_csv_auto('{PATH_MO_DIRECTORY}', header=true, all_varchar=true)")
                print(f"Представление 'mo_directory' создано из CSV файла: {PATH_MO_DIRECTORY}")
                mo_directory_created_successfully = True
            except Exception as e:
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось создать представление 'mo_directory' из CSV файла '{PATH_MO_DIRECTORY}': {e}")
        else:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось определить тип файла для справочника МО по имени: {PATH_MO_DIRECTORY}.")
        
        if not mo_directory_created_successfully:
            print("[КРИТИЧЕСКАЯ ОШИБКА] Представление 'mo_directory' НЕ создано.")
            return False
        if not all_parquet_views_created:
            print("[ПРЕДУПРЕЖДЕНИЕ] Не все представления для Parquet-файлов были успешно созданы.")
        return True
    except Exception as e:
        print(f"[ОШИБКА DuckDB] Не удалось инициализировать DuckDB на верхнем уровне: {e}")
        duckdb_con = None
        return False

def execute_duckdb_query(sql_query):
    global duckdb_con
    if duckdb_con is None: return None
    print(f"\nВыполнение SQL: {sql_query}")
    try:
        return duckdb_con.execute(sql_query).fetchdf()
    except Exception as e:
        print(f"[ОШИБКА DuckDB Query] {e}")
        return None

TABLE_CONTEXT = """
Тебе доступны следующие таблицы (представления DuckDB) и их ключевые столбцы. При генерации SQL всегда используй алиасы для таблиц (например, `p` для `population`, `ma` для `market_access`, `md` для `mo_directory`, `s` для `salary`, `mig` для `migration`, `c` для `connections`) и квалифицируй ВСЕ имена столбцов этими алиасами (например, `p.year`, `md.territory_id`, `s.value`). Это особенно важно для столбцов `territory_id` и `year`, так как они могут встречаться в нескольких таблицах.

1.  **market_access** (алиас `ma`):
    * `territory_id` (STRING): Уникальный идентификатор МО. (Используй `ma.territory_id`)
    * `market_access` (FLOAT): Индекс доступности рынков на 2024 год. (Используй `ma.market_access`)
2.  **population** (алиас `p`):
    * `territory_id` (STRING): (Используй `p.territory_id`)
    * `year` (INTEGER): Данные за 2023, 2024 гг. (Используй `p.year`)
    * `age` (STRING): Возрастная группа (например, 'Все возраста' для общего населения). (Используй `p.age`)
    * `gender` (STRING): Пол. (Используй `p.gender`)
    * `value` (INTEGER): Численность населения. (Используй `p.value`)
3.  **migration** (алиас `mig`):
    * `territory_id` (STRING): (Используй `mig.territory_id`)
    * `year` (INTEGER): Данные за 2023, 2024 гг. (Используй `mig.year`)
    * `age` (STRING): (Используй `mig.age`)
    * `gender` (STRING): (Используй `mig.gender`)
    * `value` (INTEGER): Число мигрантов. (Используй `mig.value`)
4.  **salary** (алиас `s`):
    * `territory_id` (STRING): (Используй `s.territory_id`)
    * `year` (INTEGER): С 1 кв. 2023 г. по 4 кв. 2024 г. (Используй `s.year`)
    * `period` (STRING): Квартал (например, '1 квартал', '2 квартал', '3 квартал', '4 квартал'). (Используй `s.period`)
    * `okved_name` (STRING): Наименование вида экономической деятельности (ОКВЭД). (Используй `s.okved_name`)
    * `okved_letter` (STRING): Буквенное обозначение ОКВЭД. (Используй `s.okved_letter`)
    * `value` (FLOAT): Среднемесячная заработная плата. ВАЖНО: это и есть значение зарплаты, используй `s.value` для агрегаций типа AVG. (Используй `s.value`)
5.  **connections** (алиас `c`):
    * `territory_id_x` (STRING): ID МО отправления. (Используй `c.territory_id_x`)
    * `territory_id_y` (STRING): ID МО прибытия. (Используй `c.territory_id_y`)
    * `distance` (FLOAT): Расстояние в км. (Используй `c.distance`)
6.  **mo_directory** (алиас `md`): (Справочник МО)
    * `territory_id` (STRING): Код территории. (Используй `md.territory_id`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `md.municipal_district_name`)

ПРИМЕЧАНИЕ ДЛЯ ГЕНЕРАЦИИ SQL: При JOIN `mo_directory` с другими таблицами, используй `md.territory_id = other_alias.territory_id`. Убедись, что типы данных для `territory_id` совпадают (все `territory_id` в представлениях должны быть VARCHAR, что уже учтено при создании представлений). Всегда включай `md.municipal_district_name` в SELECT, если требуются названия территорий. Для строковых литералов в SQL используй одинарные кавычки. Убедись, что любая колонка, используемая в WHERE, SELECT, GROUP BY или ORDER BY, доступна из таблиц в FROM/JOIN и правильно квалифицирована алиасом.
"""



# --- API Эндпоинт ---
@app.route('/process_query', methods=['POST'])
def process_query():
    if not duckdb_con: 
        return jsonify({"error": "База данных DuckDB не инициализирована на сервере."}), 500
    if giga_client is None:
        return jsonify({"error": "Клиент GigaChat не инициализирован на сервере."}), 500

    try:
        data = request.get_json()
        user_prompt = data.get('user_prompt')
        force_anomaly_analysis = data.get('force_analyze_anomalies', False)
        
        if not user_prompt:
            return jsonify({"error": "user_prompt не предоставлен"}), 400

        print(f"\n{'='*60}")
        print(f"📨 Получен запрос от пользователя: {user_prompt}")
        if force_anomaly_analysis:
            print("🔒 Принудительный анализ аномалий включен")

        # Запуск конвейера агентов
        print("\n🤖 Запуск мультиагентной системы...")
        formal_request = agent1_rephraser(user_prompt)
        print(f"Формальный запрос: {formal_request}")
        plan = agent2_planner(formal_request)
        print(f"План: {plan}")
        
        generated_sql_query = agent3_sql_generator(plan)
        print(f"Сгенерированный SQL: {generated_sql_query}")

        current_sql_query = generated_sql_query
        sql_validation_log = []
        
        if current_sql_query and "SELECT" in current_sql_query.upper():
            validation_result = agent_sql_validator(current_sql_query, plan)
            sql_validation_log.append({
                "query_source": "generator",
                "sql": current_sql_query,
                "validation": validation_result
            })
            print(f"Результат валидации: {validation_result}")
            if not validation_result.get("is_valid"):
                if validation_result.get("corrected_sql"):
                    print(f"Валидатор предложил исправленный SQL: {validation_result.get('corrected_sql')}")
                    current_sql_query = validation_result.get("corrected_sql")
                else:
                    # Если валидатор не смог исправить, можно либо остановиться, либо пробовать выполнить как есть
                    print(f"Валидация не пройдена, исправление не предложено: {validation_result.get('message')}")
                    # Для простоты, пока будем пытаться выполнить то, что сгенерировано, если валидатор не дал исправление
                    # В более сложном сценарии здесь можно вернуть ошибку пользователю
        
        sql_results_df = None
        sql_error_message = None
        MAX_FIX_ATTEMPTS = 2 # Максимум 1 попытка исправления + исходный запрос
        attempt_count = 0

        while attempt_count < MAX_FIX_ATTEMPTS and current_sql_query and "SELECT" in current_sql_query.upper():
            attempt_count += 1
            print(f"Попытка выполнения SQL ({attempt_count}/{MAX_FIX_ATTEMPTS}): {current_sql_query}")
            try:
                # Используем временную переменную для отлова ошибки именно от execute_duckdb_query
                temp_df = execute_duckdb_query(current_sql_query) 
                sql_results_df = temp_df # Присваиваем только если нет ошибки
                sql_error_message = None # Сбрасываем ошибку если успешно
                print("SQL выполнен успешно.")
                break # Выходим из цикла если успешно
            except Exception as e: # Ловим ошибку прямо здесь
                sql_error_message = str(e)
                print(f"[ОШИБКА DuckDB Query] {sql_error_message}")
                sql_results_df = None # Убедимся что df пустой при ошибке

            if sql_results_df is None and sql_error_message and attempt_count < MAX_FIX_ATTEMPTS:
                print("Попытка исправить SQL...")
                fixed_sql = agent_sql_fixer(current_sql_query, sql_error_message, plan)
                sql_validation_log.append({
                    "query_source": "fixer_attempt_" + str(attempt_count),
                    "original_sql": current_sql_query,
                    "error": sql_error_message,
                    "suggested_fix": fixed_sql
                })
                if fixed_sql:
                    print(f"Фиксер предложил новый SQL: {fixed_sql}")
                    current_sql_query = fixed_sql
                    # После фиксера можно снова прогнать через валидатор, если есть желание
                    validation_after_fix = agent_sql_validator(current_sql_query, plan)
                    sql_validation_log.append({
                        "query_source": "validator_after_fixer_attempt_" + str(attempt_count),
                        "sql": current_sql_query,
                        "validation": validation_after_fix
                    })
                    print(f"Результат валидации после исправления: {validation_after_fix}")
                    if not validation_after_fix.get("is_valid") and validation_after_fix.get("corrected_sql"):
                        current_sql_query = validation_after_fix.get("corrected_sql") # Применяем исправление от валидатора
                    elif not validation_after_fix.get("is_valid"):
                        print(f"Валидация после исправления не пройдена, новый SQL от валидатора не предложен: {validation_after_fix.get('message')}")
                        # Решаем, что делать - либо прервать, либо пробовать выполнить предложенное фиксером
                        # Пока что будем пробовать выполнить то, что предложил фиксер, даже если валидатор не доволен
                else:
                    print("Фиксер не смог предложить исправление.")
                    break # Выходим, если фиксер не помог
            elif sql_results_df is not None: # Если запрос выполнился после какой-то из попыток
                break
            else: # Если sql_query пустой или не SELECT, или превышены попытки
                break
        
        sql_results_str = "SQL-запрос не был выполнен или не вернул данных."
        if sql_results_df is not None:
            sql_results_str = sql_results_df.to_markdown(index=False) if not sql_results_df.empty else "Запрос SQL выполнен, но данные не найдены."
        elif sql_error_message:
             sql_results_str = f"Ошибка при выполнении SQL-запроса: {sql_error_message}"
        
        final_answer = agent4_answer_generator(sql_results_df, user_prompt)
        print("✅ Агент 4 (генерация ответа) завершен")

        response_data = {
            "formal_request": formal_request,
            "plan": plan,
            "generated_sql_query": generated_sql_query,
            "executed_sql_query": current_sql_query if sql_results_df is not None else generated_sql_query, # Показываем последний выполненный или исходный
            "sql_validation_log": sql_validation_log,
            "sql_error": sql_error_message if sql_results_df is None else None,
            "sql_results_str": sql_results_str,
            "final_answer": final_answer
        }

        print(f"✅ Обработка завершена успешно")
        print("="*60)
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Ошибка в /process_query: {e}")
        return jsonify({"error": str(e)}), 500

# --- Эндпоинт для настройки конфигурации ---
@app.route('/configure_anomaly', methods=['POST'])
def configure_anomaly():
    """Настройка параметров автоматического анализа аномалий"""
    try:
        config_updates = request.get_json()
        
        # Обновляем конфигурацию
        if 'intent_keywords' in config_updates:
            ANOMALY_AUTO_CONFIG['intent_keywords'].extend(config_updates['intent_keywords'])
        
        if 'data_criteria' in config_updates:
            ANOMALY_AUTO_CONFIG['data_criteria'].update(config_updates['data_criteria'])
        
        if 'auto_threshold' in config_updates:
            ANOMALY_AUTO_CONFIG['auto_threshold'] = config_updates['auto_threshold']
        
        return jsonify({
            "message": "Конфигурация автоматического анализа аномалий обновлена",
            "current_config": ANOMALY_AUTO_CONFIG
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Эндпоинт для получения информации о системе ---
@app.route('/system_info', methods=['GET'])
def system_info():
    """Информация о доступных агентах и конфигурации"""
    return jsonify({
        "agents": [
            {"id": 0, "name": "Intent Analyzer", "description": "Анализ намерений пользователя для определения необходимости анализа аномалий"},
            {"id": 1, "name": "Rephraser", "description": "Переформулировка запросов"},
            {"id": 2, "name": "Planner", "description": "Планирование анализа"},
            {"id": 3, "name": "SQL Generator", "description": "Генерация SQL-запросов"},
            {"id": 4, "name": "Answer Generator", "description": "Генерация ответов"},
            {"id": 5, "name": "Anomaly Detector", "description": "Обнаружение аномалий (отдельный файл)"}
        ],
        "anomaly_config": ANOMALY_AUTO_CONFIG,
        "database_status": duckdb_con is not None,
        "gigachat_status": giga_client is not None,
        "version": "2.1 - Модульная архитектура с полной отладкой"
    })

if __name__ == '__main__':
    if setup_duckdb(): 
        print("\n🚀 Мультиагентная система с модульной архитектурой запущена!")
        print("📋 Доступные агенты:")
        print("  0. 🧠 Агент анализа намерений (автоматическое определение необходимости анализа аномалий)")
        print("  1. 📝 Агент переформулировки запросов")
        print("  2. 📋 Агент планирования")
        print("  3. 💾 Агент генерации SQL")
        print("  4. 💬 Агент генерации ответов")
        print("  5. 🔍 Агент анализа аномалий (отдельный модуль)")
        print("\n⚙️ Автоматические возможности:")
        print("  - Анализ намерений пользователя для определения необходимости поиска аномалий")
        print("  - Анализ характеристик данных для принятия решения о запуске детектора аномалий")
        print("  - Мультиметодное обнаружение аномалий (Z-Score, IQR, Isolation Forest)")
        print("  - LLM-интерпретация результатов анализа аномалий")
        print("  - Подробная диагностика и отладка процесса принятия решений")
        print("\n🌐 Flask приложение готово к работе на порту 5001!")
        app.run(debug=True, port=5001) 
    else:
        print("❌ Не удалось запустить Flask приложение из-за ошибки инициализации DuckDB.")

