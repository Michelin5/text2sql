# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import duckdb
import pandas as pd
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
import numpy as np

# Импортируем наш агент анализа аномалий
from anomaly_agent import AnomalyDetectorAgent, should_analyze_anomalies_by_data

load_dotenv()

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

duckdb_con = None

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

# --- Настройка GigaChat ---
GIGA_CREDENTIALS = os.environ.get("GIGACHAT_TOKEN")
giga_client = None
anomaly_agent = None

try:
    giga_client = GigaChat(credentials=GIGA_CREDENTIALS, verify_ssl_certs=False, model="GigaChat-2-Max")
    print("Клиент GigaChat успешно инициализирован (Flask app).")
    
    # Инициализируем агента анализа аномалий
    anomaly_agent = AnomalyDetectorAgent(giga_client, verbose=True)
    print("🔍 Агент анализа аномалий инициализирован.")
    
except Exception as e:
    print(f"[ОШИБКА GigaChat Init] {e}")

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

def call_giga_api_wrapper(prompt_text, system_instruction):
    if giga_client is None: return "[ОШИБКА API] Клиент GigaChat не инициализирован."
    print(f"\nВызов GigaChat: Системная инструкция (начало): {system_instruction[:100]}... Промпт: {prompt_text[:100]}...")
    try:
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_text)]
        res = giga_client.invoke(messages)
        print(f"GigaChat ответ (начало): {res.content[:100]}...")
        return res.content.strip()
    except Exception as e:
        print(f"Ошибка вызова GigaChat API: {e}")
        return f"[ОШИБКА API] {e}"

# --- АГЕНТ 0: Анализатор намерений ---
def agent0_intent_analyzer(user_prompt):
    """Агент для анализа намерений и определения необходимости анализа аномалий"""
    
    system_instruction = """
    Ты — эксперт по анализу намерений пользователя. Определи, нужен ли анализ аномалий для данного запроса.
    
    Анализ аномалий НУЖЕН если запрос содержит:
    - Слова: "аномалии", "выбросы", "необычные", "странные", "подозрительные", "отклонения"
    - Просьбы найти "максимальные", "минимальные", "экстремальные" значения
    - Вопросы о "качестве данных", "проблемах", "ошибках"  
    - Сравнения регионов/территорий (могут выявить аномальные значения)
    - Анализ зарплат, миграции, населения (часто содержат аномалии)
    - Временной анализ, тренды (могут показать аномалии)
    - Слова "анализ", "исследование", "различия", "распределение", "выявить"
    - Вопросы типа "какие регионы отличаются", "где больше всего", "где меньше всего"
    
    Анализ аномалий НЕ НУЖЕН если запрос:
    - Простой поиск конкретных значений
    - Подсчет количества записей
    - Получение справочной информации
    - Простые фактические вопросы без сравнений
    
    Ответь ТОЛЬКО "true" или "false".
    """
    
    try:
        response = call_giga_api_wrapper(user_prompt, system_instruction)
        print(f"🔍 Ответ LLM для анализа намерений: '{response}'")  # Отладка
        response_lower = response.lower().strip()
        
        if "true" in response_lower:
            print("✅ LLM рекомендует анализ аномалий")
            return True
        elif "false" in response_lower:
            print("❌ LLM НЕ рекомендует анализ аномалий")
            return False
        else:
            print(f"⚠️ Неопределенный ответ LLM: '{response}', используем резервный метод")
            return _fallback_anomaly_detection(user_prompt)
    except Exception as e:
        print(f"❌ Ошибка в агенте анализа намерений: {e}")
        return _fallback_anomaly_detection(user_prompt)

def _fallback_anomaly_detection(user_prompt):
    """Резервный метод определения необходимости анализа аномалий"""
    user_prompt_lower = user_prompt.lower()
    found_keywords = [keyword for keyword in ANOMALY_AUTO_CONFIG["intent_keywords"] if keyword in user_prompt_lower]
    if found_keywords:
        print(f"🔍 Резервный метод: найдены ключевые слова: {found_keywords}")
        return True
    else:
        print("🔍 Резервный метод: ключевые слова не найдены")
        return False

# --- Существующие агенты ---
def agent1_rephraser(user_prompt):
    system_instruction = "Ты — ИИ-ассистент. Переформулируй запрос пользователя в формальный, четкий вопрос для анализа данных на русском языке. Вывод должен содержать ТОЛЬКО переформулированный вопрос."
    return call_giga_api_wrapper(user_prompt, system_instruction)

def agent2_planner(formal_prompt):
    system_instruction = (
        "Ты — ИИ-планировщик. На основе формального запроса создай краткий, пронумерованный пошаговый план на русском языке. "
        "Используй TABLE_CONTEXT. Вывод должен содержать ТОЛЬКО план. НЕ ВКЛЮЧАЙ SQL."
        f"\n\n{TABLE_CONTEXT}"
    )
    return call_giga_api_wrapper(formal_prompt, system_instruction)

def agent3_sql_generator(plan):
    system_instruction = (
        "Ты — эксперт по генерации SQL. На основе плана и TABLE_CONTEXT сгенерируй SQL-запрос. "
        "ВАЖНО: ВСЕГДА используй алиасы для таблиц (например, `p` для `population`, `ma` для `market_access`, `md` для `mo_directory`, `s` для `salary`) и ВСЕГДА квалифицируй КАЖДОЕ имя столбца этим алиасом (например, `p.year`, `md.territory_id`, `s.value` для зарплаты). Это КРИТИЧЕСКИ важно для избежания ошибок неоднозначности, особенно для `territory_id` и `year`. "
        "КРАЙНЕ ВАЖНО: Если план включает извлечение `territory_id` или данных, связанных с территориями, ты ОБЯЗАН включить в SELECT столбец `md.municipal_district_name` и выполнить JOIN с представлением `mo_directory` (алиас `md`) по `md.territory_id = relevant_table_alias.territory_id` для отображения названий МО. "
        "При указании условий на колонку `year` (например, `WHERE p.year = 2024`), убедись, что таблица, к которой относится `year` (например, `population` с алиасом `p`), корректно включена в `FROM` или `JOIN` часть запроса и колонка `year` квалифицирована (например, `p.year`). "
        "Для таблицы `salary` (алиас `s`), фактическое значение зарплаты находится в колонке `s.value`. При расчете среднего используй `AVG(s.value)`. "
        "Вывод должен быть ТОЛЬКО SQL-запрос в блоке ``````."
        f"\n\n{TABLE_CONTEXT}"
    )
    sql_query_block = call_giga_api_wrapper(plan, system_instruction)
    if sql_query_block and "```" in sql_query_block:
        try:
            # Try to extract SQL from ```sql ... ``` or just ``` ... ```
            if "```sql" in sql_query_block:
                return sql_query_block.split("```sql")[1].split("```")[0].strip()
            else:
                return sql_query_block.split("```")[1].split("```")[0].strip()
        except IndexError:
            return sql_query_block.strip()
    elif sql_query_block and sql_query_block.strip().upper().startswith("SELECT"):
        return sql_query_block.strip()
    return sql_query_block

def agent4_answer_generator(sql_result_df, initial_user_prompt):
    if sql_result_df is None: 
        return "К сожалению, не удалось получить данные из-за ошибки SQL."
    if sql_result_df.empty: 
        return "По вашему запросу данные не найдены."
    
    data_as_string = sql_result_df.to_markdown(index=False) if not sql_result_df.empty else "Данные отсутствуют."
    system_instruction = (
        "Ты — ИИ-ассистент, объясняющий результаты анализа данных. "
        "Тебе дан исходный вопрос пользователя и данные (результат SQL в Markdown). "
        "Сформулируй ясный ответ на русском языке. Если видишь NaN, укажи, что данные отсутствуют."
    )
    prompt_for_llm = (
        f"Исходный вопрос пользователя: \"{initial_user_prompt}\"\n\n"
        f"Данные для ответа (Markdown):\n{data_as_string}\n\nОтвет:"
    )
    return call_giga_api_wrapper(prompt_for_llm, system_instruction)

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

        # ШАГ 0: Анализ намерений для определения необходимости анализа аномалий
        auto_anomaly_needed = agent0_intent_analyzer(user_prompt)
        print(f"🧠 Агент анализа намерений: анализ аномалий {'НУЖЕН' if auto_anomaly_needed else 'НЕ НУЖЕН'}")

        # Запуск конвейера агентов
        print("\n🤖 Запуск мультиагентной системы...")
        formal_request = agent1_rephraser(user_prompt)
        print("✅ Агент 1 (переформулировка) завершен")
        
        plan = agent2_planner(formal_request)
        print("✅ Агент 2 (планирование) завершен")
        
        sql_query = agent3_sql_generator(plan)
        print("✅ Агент 3 (генерация SQL) завершен")
        
        sql_results_df = None
        sql_results_str = "SQL-запрос не был выполнен или не вернул данных."
        anomaly_analysis = None
        data_based_anomaly_needed = False
        
        if sql_query and "SELECT" in sql_query.upper() and duckdb_con:
            sql_results_df = execute_duckdb_query(sql_query)
            if sql_results_df is not None:
                sql_results_str = sql_results_df.to_markdown(index=False) if not sql_results_df.empty else "Запрос SQL выполнен, но данные не найдены."
                
                # ШАГ: Дополнительная проверка на основе данных
                data_based_anomaly_needed = should_analyze_anomalies_by_data(
                    sql_results_df, sql_query, ANOMALY_AUTO_CONFIG["data_criteria"]
                )
                print(f"📊 Анализ данных: анализ аномалий {'РЕКОМЕНДОВАН' if data_based_anomaly_needed else 'НЕ РЕКОМЕНДОВАН'}")
                
                # Диагностика запуска агента аномалий
                print(f"\n🔍 Диагностика запуска агента аномалий:")
                print(f"   force_anomaly_analysis: {force_anomaly_analysis}")
                print(f"   auto_anomaly_needed: {auto_anomaly_needed}")
                print(f"   data_based_anomaly_needed: {data_based_anomaly_needed}")
                print(f"   sql_results_df пустой: {sql_results_df is None or sql_results_df.empty}")
                print(f"   anomaly_agent доступен: {anomaly_agent is not None}")
                
                # Окончательное решение об анализе аномалий
                should_analyze = force_anomaly_analysis or auto_anomaly_needed or data_based_anomaly_needed
                print(f"   Финальное решение should_analyze: {should_analyze}")
                
                if should_analyze and not sql_results_df.empty and anomaly_agent:
                    print("\n🔍 ✅ Все условия выполнены - запускаю агента анализа аномалий...")
                    anomaly_analysis = anomaly_agent.detect_anomalies(sql_results_df, user_prompt)
                    
                    # Логируем результат
                    if anomaly_analysis.get("anomalies_found"):
                        print(f"⚠️ Обнаружены аномалии в {anomaly_analysis.get('analyzed_columns', [])} столбцах")
                    else:
                        print("✅ Значительных аномалий не обнаружено")
                else:
                    print("\n🔍 ❌ Условия для запуска агента НЕ выполнены:")
                    if not should_analyze:
                        print("     - should_analyze = False")
                    if sql_results_df is None or sql_results_df.empty:
                        print("     - sql_results_df пустой")
                    if not anomaly_agent:
                        print("     - anomaly_agent не инициализирован")
            else:
                sql_results_str = "Ошибка при выполнении SQL-запроса (подробности см. в логах сервера)."
        
        final_answer = agent4_answer_generator(sql_results_df, user_prompt)
        print("✅ Агент 4 (генерация ответа) завершен")
        
        # Добавляем результаты анализа аномалий к финальному ответу
        if anomaly_analysis and anomaly_analysis.get("anomalies_found"):
            final_answer += f"\n\n🚨 **Автоматический анализ аномалий:**\n{anomaly_analysis['message']}"

        response_data = {
            "formal_request": formal_request,
            "plan": plan,
            "sql_query": sql_query,
            "sql_results_str": sql_results_str,
            "final_answer": final_answer,
            "auto_anomaly_analysis": {
                "intent_based": auto_anomaly_needed,
                "data_based": data_based_anomaly_needed,
                "executed": anomaly_analysis is not None,
                "forced": force_anomaly_analysis
            }
        }
        
        # Добавляем результаты анализа аномалий в ответ
        if anomaly_analysis:
            response_data["anomaly_analysis"] = anomaly_analysis

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
        "anomaly_agent_status": anomaly_agent is not None,
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

