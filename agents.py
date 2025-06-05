import json
from giga_wrapper import call_giga_api_wrapper

# --- TABLE_CONTEXT (copy from app.py) ---
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
            return sql_query_block.split("```sql")[1].split("```", 1)[0].strip()
        except IndexError: return sql_query_block
    elif sql_query_block and sql_query_block.strip().upper().startswith("SELECT"):
        return sql_query_block.strip()
    return sql_query_block

def agent_sql_validator(sql_query, plan):
    """
    Агент для валидации SQL-запроса.
    Проверяет синтаксис, соответствие схеме и базовую логику.
    Возвращает словарь: {"is_valid": True/False, "message": "...", "corrected_sql": "..."}
    """
    system_instruction = (
        "Ты — ИИ-валидатор SQL-запросов. Тебе предоставлен SQL-запрос, первоначальный план и информация о схеме таблиц (TABLE_CONTEXT). "
        "Твоя задача — проверить SQL-запрос на:\n"
        "1. Корректность синтаксиса SQL.\n"
        "2. Соответствие именам таблиц и столбцов, указанным в TABLE_CONTEXT.\n"
        "3. Правильное использование алиасов таблиц и квалификацию всех столбцов алиасами.\n"
        "4. Логическое соответствие запроса плану. Например, если в плане есть 'найти среднюю зарплату', в SQL должно быть AVG().\n"
        "5. Если в плане указано получение названий МО, в SQL должен быть JOIN с 'mo_directory' и SELECT 'md.municipal_district_name'.\n"
        "6. Для таблицы 'salary' (алиас 's'), агрегации должны применяться к 's.value'.\n"
        "7. Для строковых литералов в SQL должны использоваться одинарные кавычки.\n"
        "Если запрос полностью корректен, верни: `{\"is_valid\": true, \"message\": \"SQL-запрос корректен.\"}`.\n"
        "Если найдены ошибки, верни: {\"is_valid\": false, \"message\": \"Обнаружены следующие ошибки: [описание ошибок]\", \"corrected_sql\": \"[предложенный исправленный SQL, если можешь его дать, иначе пустая строка]\"}`. "
        "Вывод должен быть ТОЛЬКО JSON объектом."
        f"\n\nПлан:\n{plan}\n\n{TABLE_CONTEXT}"
    )
    prompt_for_llm = f"SQL для проверки:\n```sql\n{sql_query}\n```"
    response_str = call_giga_api_wrapper(prompt_for_llm, system_instruction)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        print(f"[ОШИБКА ВАЛИДАТОРА] Не удалось распарсить JSON от валидатора: {response_str}")
        return {"is_valid": False, "message": "Ошибка валидатора: невалидный JSON ответ.", "corrected_sql": None}


def agent_sql_fixer(failed_sql_query, error_message, plan):
    """
    Агент для исправления SQL-запроса после неудачного выполнения.
    """
    system_instruction = (
        "Ты — ИИ-эксперт по исправлению SQL-запросов. Тебе дан SQL-запрос, который не удалось выполнить, "
        "сообщение об ошибке от базы данных, первоначальный план и информация о схеме (TABLE_CONTEXT). "
        "Проанализируй ошибку и предложи исправленный SQL-запрос. "
        "Обрати особое внимание на:\n"
        "1. Синтаксические ошибки, указанные в сообщении.\n"
        "2. Неправильные имена таблиц или столбцов (сверься с TABLE_CONTEXT).\n"
        "3. Отсутствие или неправильное использование алиасов (все столбцы должны быть квалифицированы).\n"
        "4. Ошибки в JOIN (неправильные условия, отсутствующие таблицы).\n"
        "5. Несоответствие типов данных.\n"
        "6. Логические ошибки, из-за которых мог произойти сбой (например, деление на ноль, если это можно предположить из контекста ошибки).\n"
        "Вывод должен быть ТОЛЬКО исправленным SQL-запросом в блоке ```sql ... ```. Если не можешь предложить исправление, верни пустой блок ```sql\n```."
        f"\n\nПервоначальный план:\n{plan}\n\nСообщение об ошибке:\n{error_message}\n\n{TABLE_CONTEXT}"
    )
    prompt_for_llm = f"Неудавшийся SQL-запрос:\n```sql\n{failed_sql_query}\n```"
    corrected_sql_block = call_giga_api_wrapper(prompt_for_llm, system_instruction)
    
    if corrected_sql_block and "```sql" in corrected_sql_block:
        try:
            corrected_sql = corrected_sql_block.split("```sql")[1].split("```", 1)[0].strip()
            return corrected_sql if corrected_sql else None # Возвращаем None если блок пустой
        except IndexError:
            return None 
    return None

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