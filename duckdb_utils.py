import duckdb
import pandas as pd
import os

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