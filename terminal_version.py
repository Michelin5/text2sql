from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import pandas as pd
import sqlite3
import json
import sqliteschema
import os
import argparse
import sys
from dotenv import load_dotenv

class SQLAssistant:
    def __init__(self, data_path='data/2_bdmo_population.parquet', db_path='population.db'):
        self.data_path = data_path
        self.db_path = db_path
        self.llm = None
        self.system_prompt = None
        
    def setup_database(self):
        """Настройка базы данных"""
        print("📊 Загрузка данных из parquet файла...")
        try:
            population = pd.read_parquet(self.data_path)
            print(f"✅ Загружено {len(population)} строк")
            print(population.head())
            
            print("💾 Создание SQLite базы данных...")
            connection = sqlite3.connect(self.db_path)
            population.to_sql('population', connection, if_exists='replace', index=False)
            connection.close()
            print("✅ База данных создана успешно")
            
            return population
            
        except Exception as e:
            print(f"❌ Ошибка при настройке базы данных: {e}")
            return None
    
    def get_column_values_summary(self, df):
        """Получение уникальных значений в столбцах"""
        summary = {}
        for column in df.columns:
            unique_values = df[column].dropna().unique()
            if len(unique_values) <= 50:
                summary[column] = sorted(map(str, unique_values))
        return summary
    
    def setup_llm(self):
        """Настройка LLM"""
        load_dotenv()
        giga_token = os.getenv("GIGACHAT_TOKEN")
        if giga_token is None:
            raise ValueError("❌ Переменная окружения GIGACHAT_TOKEN не установлена!")
        
        self.llm = GigaChat(
            credentials=giga_token, 
            verify_ssl_certs=False,
            model="GigaChat-2-Max",
            temperature=0.1
        )
        print("🤖 GigaChat настроен успешно")
    
    def create_system_prompt(self, population):
        """Создание системного промпта"""
        extractor = sqliteschema.SQLiteSchemaExtractor(self.db_path)
        schema = extractor.fetch_database_schema_as_dict()
        column_values_summary = self.get_column_values_summary(population)
        
        self.system_prompt = f"""
# Система генерации SQL-запросов для SQLite

Вы являетесь экспертом в генерации SQL-запросов для базы данных SQLite.

## Правила:
- Используйте только таблицу population
- Используйте только существующие поля из схемы
- Избегайте SELECT * — всегда указывайте конкретные столбцы
- Генерируйте только валидный SQL без пояснений
- Используйте DISTINCT где необходимо

SQL-схема:
{json.dumps(schema, indent=2)}

Значения столбцов:
{json.dumps(column_values_summary, indent=2)}
"""
    
    def execute_sql_query(self, sql: str):
        """Выполнение SQL запроса"""
        try:
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query(sql, conn)
            conn.close()
            return result
        except Exception as e:
            return f"❌ Ошибка при выполнении SQL: {e}"
    
    def generate_sql(self, user_question: str) -> str:
        """Генерация SQL запроса"""
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_question)
        ])
        sql = response.content.strip()
        return sql
    
    def ask_question(self, user_question: str):
        """Обработка вопроса пользователя"""
        print(f"\n💬 Вопрос: {user_question}")
        
        sql = self.generate_sql(user_question)
        print(f"\n📜 Сгенерированный SQL:\n{sql}\n")
        
        if not sql:
            return "❌ Не удалось сгенерировать SQL-запрос."
        
        result = self.execute_sql_query(sql)
        
        if isinstance(result, pd.DataFrame):
            if result.empty:
                print("🔍 Результаты не найдены.")
                return None
            else:
                print("📊 Результаты:")
                print(result.to_string(index=False))
                return result
        else:
            print(result)
            return None

def main():
    """Основная функция для работы в терминале"""
    parser = argparse.ArgumentParser(description='SQL Assistant для анализа данных населения')
    parser.add_argument('--data', default='data/2_bdmo_population.parquet', 
                       help='Путь к parquet файлу')
    parser.add_argument('--db', default='population.db', 
                       help='Путь к SQLite базе данных')
    parser.add_argument('--setup-only', action='store_true', 
                       help='Только настроить базу данных и выйти')
    
    args = parser.parse_args()
    
    # Инициализация ассистента
    assistant = SQLAssistant(args.data, args.db)
    
    # Настройка базы данных
    population = assistant.setup_database()
    if population is None:
        sys.exit(1)
    
    if args.setup_only:
        print("✅ База данных настроена. Выход.")
        return
    
    # Настройка LLM
    try:
        assistant.setup_llm()
        assistant.create_system_prompt(population)
    except Exception as e:
        print(f"❌ Ошибка настройки LLM: {e}")
        sys.exit(1)
    
    print("\n🚀 SQL Assistant готов к работе!")
    print("Введите ваши вопросы о данных населения (или 'exit' для выхода):")
    print("Примеры вопросов:")
    print("- Покажи общую численность населения по регионам")
    print("- Какие города имеют население больше 1 миллиона?")
    print("- Покажи динамику населения за последние годы")
    
    # Интерактивный режим
    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()
            
            if question.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
            
            if not question:
                continue
                
            assistant.ask_question(question)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
