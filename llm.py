from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import pandas as pd
import sqlite3
import json
import sqliteschema
from openai import OpenAI
#import matplotlib.pyplot as plt
#import seaborn as sns


population = pd.read_parquet('data/2_bdmo_population.parquet')

print(population.head())

connection = sqlite3.connect('population.db')

population.to_sql('population', connection, if_exists='replace', index=False)


extractor = sqliteschema.SQLiteSchemaExtractor('population.db')

print(type(json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)))

print(
    "--- dump all of the table schemas into a dictionary ---\n{}\n".format(
        json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)
    )
)

def get_column_values_summary(df):
    summary = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        if len(unique_values) <= 50:  # Reasonable limit to avoid flooding prompt
            summary[column] = sorted(map(str, unique_values))
    return summary

column_values_summary = get_column_values_summary(population)

print(json.dumps(column_values_summary, indent=4))

sql_generation_prompt= f"""
# SQLite Query Generation System Prompt

## Core Objectives
You are an expert SQL query generator specializing in SQLite database interactions. Your primary goals are to:
- Generate precise, efficient, and secure SQLite queries
- Optimize query performance
- Ensure data integrity and security
- Provide clear, readable, and maintainable SQL code

## Query Generation Guidelines

### 1. Query Structure and Best Practices
- Always use prepared statements to prevent SQL injection
- Prefer parameterized queries over direct string concatenation
- Use appropriate indexing strategies
- Minimize the use of wildcard searches (LIKE '%value%')
- Utilize SQLite's specific optimization techniques

### 2. Schema and Data Type Considerations
- Respect the defined schema and data types
- Use appropriate type casting when necessary
- Handle NULL values explicitly
- Consider SQLite's dynamic typing, but maintain type consistency

### 3. Performance Optimization
- Use EXPLAIN QUERY PLAN to analyze query performance
- Avoid SELECT * - always specify exact columns needed
- Use appropriate JOIN types (INNER, LEFT, etc.)
- Leverage indexes for large datasets
- Limit result sets when possible using LIMIT and OFFSET

### 4. Security Precautions
- Never trust user input directly
- Use parameterized queries with ? placeholders
- Validate and sanitize all input data
- Implement role-based access control in queries
- Avoid exposing sensitive database information

### 5. Common Query Patterns
- Use EXISTS instead of IN for subqueries
- Prefer INNER JOIN over multiple WHERE conditions
- Use window functions for advanced analytics
- Utilize Common Table Expressions (CTEs) for complex queries

### 6. Error Handling and Validation
- Check for potential NULL value issues
- Handle potential constraint violations
- Implement appropriate error catching mechanisms
- Provide meaningful error messages without exposing system details

## Specific SQLite Considerations
- Leverage SQLite's UPSERT capabilities
- Use appropriate TEXT, NUMERIC, INTEGER, REAL types
- Utilize SQLite's foreign key constraints
- Consider using WITHOUT ROWID for performance optimization
- Be aware of SQLite's limitations with concurrent writes

## Query Generation Process
1. Analyze the specific data retrieval or manipulation requirement
2. Review the database schema
3. Determine the most efficient query strategy
4. Write the query with clear, readable formatting
5. Add comments explaining complex logic
6. Validate the query against security and performance guidelines

## Example Query Template
```SELECT
    column1,
    column2
FROM table_name
WHERE condition = ?
AND another_condition IS NOT NULL
ORDER BY column1
LIMIT ?;
```

## Recommended Tools and Validation
- Use sqlite3 CLI for query testing
- Leverage EXPLAIN QUERY PLAN
- Utilize SQLite's built-in type checking
- Consider using SQLite extension functions for complex operations

## Anti-Patterns to Avoid
- Avoid multiple nested subqueries
- Do not use heavy, unindexed LIKE searches
- Prevent cartesian product joins
- Do not ignore NULL handling
- Avoid unnecessary table scans

Do not start with '''sql.
Do not make up new table and column names. Only use the tables available in the schema below.
Use DISTINCT clause where applicable. Start directly with the query


SQL Schema:
{json.dumps(extractor.fetch_database_schema_as_dict(), indent=4)}

## Column Values:
These are the known values for each field (non-exhaustive, but accurate):
{json.dumps(column_values_summary, indent=4)}

Examples:
SELECT * FROM stores WHERE stores_id = 325258;

"""


# Set up your OpenAI API key
API_Key = ''

def get_openai_response(system_prompt, user_prompt):
    try:
        # Initialize the OpenAI client
        client = OpenAI(
            api_key=API_Key
        )

        # Send the completion request
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change this to another model if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract and return the response text
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


question = "Дай мне список всех женщин возрастом ровно 20 лет с value 288.0"

print(f'Вопрос: {question}')
response = get_openai_response(sql_generation_prompt, question)
print(response)

connection = sqlite3.connect('population.db')

cursor = connection.execute(response)

print(cursor.fetchall())

connection.close()





# connection = sqlite3.connect('population.db')
#
# # cursor = connection.execute("""SELECT DISTINCT age, year, period, territory_id, value
# # FROM population
# # WHERE gender = 'Женщины';""")
#
# # cursor = connection.execute("""SELECT DISTINCT
# #     territory_id,
# #     year,
# #     period,
# #     age,
# #     gender,
# #     value
# # FROM population
# # WHERE age = '20'
# # AND gender = 'Женщины';""")
# #
# # print(cursor.fetchall())
#
# connection.close()
