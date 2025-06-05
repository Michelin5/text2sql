# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Для разрешения запросов с другого порта (если фронтенд отдельно)
import duckdb
import pandas as pd
import os

from giga_wrapper import giga_client, call_giga_api_wrapper

# --- Конфигурация Flask ---
app = Flask(__name__)
CORS(app) # Разрешает CORS-запросы. Полезно при локальной разработке.

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
            # Решите, является ли это критической ошибкой для вашего приложения
            # return False 
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



# --- Агенты ---
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
        "Вывод должен быть ТОЛЬКО SQL-запрос в блоке ```sql ... ```."
        f"\n\n{TABLE_CONTEXT}"
    )
    sql_query_block = call_giga_api_wrapper(plan, system_instruction)
    if sql_query_block and "```sql" in sql_query_block:
        try:
            return sql_query_block.split("```sql")[1].split("```")[0].strip()
        except IndexError: return sql_query_block 
    elif sql_query_block and sql_query_block.strip().upper().startswith("SELECT"):
        return sql_query_block.strip()
    return sql_query_block

def agent4_answer_generator(sql_result_df, initial_user_prompt):
    if sql_result_df is None: return "К сожалению, не удалось получить данные из-за ошибки SQL."
    if sql_result_df.empty: return "По вашему запросу данные не найдены."
    
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
        if not user_prompt:
            return jsonify({"error": "user_prompt не предоставлен"}), 400

        print(f"\nПолучен запрос от пользователя: {user_prompt}")

        # Запуск конвейера агентов
        formal_request = agent1_rephraser(user_prompt)
        plan = agent2_planner(formal_request)
        sql_query = agent3_sql_generator(plan)
        
        sql_results_df = None
        sql_results_str = "SQL-запрос не был выполнен или не вернул данных."
        if sql_query and "SELECT" in sql_query.upper() and duckdb_con:
            sql_results_df = execute_duckdb_query(sql_query)
            if sql_results_df is not None:
                sql_results_str = sql_results_df.to_markdown(index=False) if not sql_results_df.empty else "Запрос SQL выполнен, но данные не найдены."
            else:
                sql_results_str = "Ошибка при выполнении SQL-запроса (подробности см. в логах сервера)."
        
        final_answer = agent4_answer_generator(sql_results_df, user_prompt)

        return jsonify({
            "formal_request": formal_request,
            "plan": plan,
            "sql_query": sql_query,
            "sql_results_str": sql_results_str,
            "final_answer": final_answer
        })

    except Exception as e:
        print(f"Ошибка в /process_query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if setup_duckdb(): 
        app.run(debug=True, port=5001) 
    else:
        print("Не удалось запустить Flask приложение из-за ошибки инициализации DuckDB.")

