from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import pandas as pd
import sqlite3
import json
import sqliteschema
from openai import OpenAI
import os
#import matplotlib.pyplot as plt
#import seaborn as sns

#Загружаем данные из parquet файла
population = pd.read_parquet('data/2_bdmo_population.parquet')

print(population.head())

connection = sqlite3.connect('population.db')

population.to_sql('population', connection, if_exists='replace', index=False)

# Схема БД
extractor = sqliteschema.SQLiteSchemaExtractor('population.db')

print(type(json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)))

print(
    "--- dump all of the table schemas into a dictionary ---\n{}\n".format(
        json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)
    )
)

# Функция для получения уникальных значений в столбцах
def get_column_values_summary(df):
    summary = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        if len(unique_values) <= 50:  # Reasonable limit to avoid flooding prompt
            summary[column] = sorted(map(str, unique_values))
    return summary

column_values_summary = get_column_values_summary(population)

print(json.dumps(column_values_summary, indent=4))


system_prompt= f"""
# # Система генерации SQL-запросов для SQLite

## Основные цели
Вы являетесь экспертом в генерации SQL-запросов, специализирующимся на взаимодействии с базами данных SQLite. Ваши основные задачи:
- Генерировать точные, эффективные и безопасные запросы SQLite
- Оптимизировать производительность запросов
- Обеспечивать целостность и безопасность данных
- Предоставлять понятный, читаемый и поддерживаемый SQL-код

## Руководство по генерации запросов

### 1. Структура запросов и лучшие практики
- Всегда используйте подготовленные выражения (prepared statements) для предотвращения SQL-инъекций
- Предпочитайте параметризованные запросы вместо прямой конкатенации строк
- Используйте подходящие стратегии индексации
- Минимизируйте использование поиска по шаблону с подстановочными знаками (LIKE '%значение%')
- Применяйте специфические техники оптимизации SQLite

### 2. Учет схемы и типов данных
- Уважайте определенную схему и типы данных
- Используйте приведение типов, когда это необходимо
- Явно обрабатывайте значения NULL
- Учитывайте динамическую типизацию SQLite, но сохраняйте согласованность типов

### 3. Оптимизация производительности
- Используйте EXPLAIN QUERY PLAN для анализа производительности запросов
- Избегайте SELECT * — всегда указывайте конкретные нужные столбцы
- Применяйте подходящие типы JOIN (INNER, LEFT и т.д.)
- Используйте индексы для больших наборов данных
- Ограничивайте результирующие наборы с помощью LIMIT и OFFSET, когда это возможно

### 4. Меры безопасности
- Никогда не доверяйте пользовательскому вводу напрямую
- Используйте параметризованные запросы с заполнителями ?
- Проверяйте и очищайте все входные данные
- Реализуйте управление доступом на основе ролей в запросах
- Избегайте раскрытия конфиденциальной информации о базе данных

### 5. Типичные шаблоны запросов
- Используйте EXISTS вместо IN для подзапросов
- Предпочитайте INNER JOIN множественным условиям WHERE
- Применяйте оконные функции для сложной аналитики
- Используйте общие табличные выражения (CTE) для сложных запросов

### 6. Обработка ошибок и валидация
- Проверяйте возможные проблемы с NULL-значениями
- Обрабатывайте потенциальные нарушения ограничений
- Реализуйте подходящие механизмы перехвата ошибок
- Предоставляйте понятные сообщения об ошибках, не раскрывая системные детали

## Особенности SQLite
- Используйте возможности UPSERT в SQLite
- Применяйте подходящие типы TEXT, NUMERIC, INTEGER, REAL
- Используйте ограничения внешних ключей SQLite
- Рассмотрите использование WITHOUT ROWID для оптимизации производительности
- Учитывайте ограничения SQLite при одновременной записи

## Процесс генерации запросов
1. Проанализируйте конкретные требования к извлечению или изменению данных
2. Изучите схему базы данных
3. Определите наиболее эффективную стратегию запроса
4. Напишите запрос с четким и читаемым форматированием
5. Добавьте комментарии, объясняющие сложную логику
6. Проверьте запрос на соответствие рекомендациям по безопасности и производительности

## Пример шаблона запроса
SELECT
column1,
column2
FROM table_name
WHERE condition = ?
AND another_condition IS NOT NULL
ORDER BY column1
LIMIT ?;

## Рекомендуемые инструменты и валидация
- Используйте sqlite3 CLI для тестирования запросов
- Применяйте EXPLAIN QUERY PLAN
- Используйте встроенную проверку типов SQLite
- Рассмотрите использование расширений SQLite для сложных операций

## Антипаттерны, которых следует избегать
- Избегайте множественных вложенных подзапросов
- Не используйте тяжелые, неиндексированные поиски LIKE
- Предотвращайте декартовы произведения в соединениях
- Не игнорируйте обработку NULL
- Избегайте ненужного полного сканирования таблиц

Не начинайте с '''sql.
Не придумывайте новые имена таблиц и столбцов. Используйте только таблицы, доступные в схеме ниже.
Используйте предложение DISTINCT, где это применимо. Начинайте непосредственно с запроса.

SQL-схема:
{json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)}

## Значения столбцов:
Это известные значения для каждого поля (не исчерпывающие, но точные):
{json.dumps(column_values_summary, indent=4)}

Примеры:
SELECT * FROM stores WHERE stores_id = 325258;
"""

# Set up your OpenAI API key
giga_token = os.getenv("GIGACHAT_TOKEN")
if giga_token is None:
    raise ValueError("Переменная окружения GIGACHAT_TOKEN не установлена!")

API_Key = giga_token


# Set up your OpenAI API key
giga_token = os.getenv("GIGACHAT_TOKEN")
if giga_token is None:
    raise ValueError("Переменная окружения GIGACHAT_TOKEN не установлена!")

API_Key = giga_token

llm = GigaChat(
    credentials=giga_token, 
    verify_ssl_certs=False,
    model="GigaChat-2-Max"  
)


def execute_sql_query(sql: str):
    try:
        conn = sqlite3.connect('population.db')
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result
    except Exception as e:
        return f"❌ Ошибка при выполнении SQL: {e}"

def execute_sql_query(sql: str):
    try:
        conn = sqlite3.connect('population.db')
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result
    except Exception as e:
        return f"❌ Ошибка при выполнении SQL: {e}"
    


def generate_sql(user_question: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ])
    sql = response.content.strip()
    print(f"\n📜 Сгенерированный SQL-запрос:\n{sql}\n")
    return sql




def ask_gigachat(user_question: str):
    print(f"💬 Вопрос пользователя: {user_question}")
    
    sql = generate_sql(user_question)
    
    if not sql:
        return "❌ Не удалось сгенерировать SQL-запрос."
    
    result = execute_sql_query(sql)
    
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "🔍 Результаты не найдены."
        else:
            return result
    else:
        return result