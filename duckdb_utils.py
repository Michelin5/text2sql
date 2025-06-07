import duckdb
import os

PATH_MARKET_ACCESS = 'data/1_market_access.parquet'
PATH_POPULATION = 'data/2_bdmo_population.parquet'
PATH_MIGRATION = 'data/3_bdmo_migration.parquet'
PATH_SALARY = 'data/4_bdmo_salary.parquet'
PATH_CONNECTIONS = 'data/5_connection.parquet'

duckdb_con = None

def setup_duckdb():
    global duckdb_con
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

        if not all_parquet_views_created:
            print("[ПРЕДУПРЕЖДЕНИЕ] Не все представления для Parquet-файлов были успешно созданы.")
        return duckdb_con
    except Exception as e:
        print(f"[ОШИБКА DuckDB] Не удалось инициализировать DuckDB на верхнем уровне: {e}")
        duckdb_con = None
        return None

async def execute_duckdb_query(connection, sql_query):
    """Execute a SQL query using the provided DuckDB connection"""
    if connection is None: return None
    print(f"\nВыполнение SQL: {sql_query}")
    try:
        # DuckDB operations are CPU-bound, so we run them in a thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: connection.execute(sql_query).fetchdf())
    except Exception as e:
        print(f"[ОШИБКА DuckDB Query] {e}")
        return None 