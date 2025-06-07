import json
import numpy as np
import pandas as pd
from giga_wrapper import call_giga_api_wrapper

# --- ОБНОВЛЕННЫЙ TABLE_CONTEXT ---
TABLE_CONTEXT = """
Тебе доступны следующие таблицы (представления DuckDB) и их ключевые столбцы. При генерации SQL всегда используй алиасы для таблиц (например, `p` для `population`, `ma` для `market_access`, `s` для `salary`, `mig` для `migration`, `c` для `connections`) и квалифицируй ВСЕ имена столбцов этими алиасами (например, `p.year`, `md.territory_id`, `s.value`). Это особенно важно для столбцов `territory_id` и `year`, так как они могут встречаться в нескольких таблицах.

1.  **market_access** (алиас `ma`):
    * `territory_id` (STRING): Уникальный идентификатор МО. (Используй `ma.territory_id`)
    * `market_access` (FLOAT): Индекс доступности рынков на 2024 год. (Используй `ma.market_access`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `ma.municipal_district_name`)
2.  **population** (алиас `p`):
    * `territory_id` (STRING): (Используй `p.territory_id`)
    * `year` (INTEGER): Данные за 2023, 2024 гг. (Используй `p.year`)
    * `age` (STRING): Возрастная группа (например, 'Все возраста' для общего населения). (Используй `p.age`). Все значения столбца p.age это числа, например: '20', '1', '15', '75'
    * `gender` (STRING): Пол. (Используй `p.gender`). Все уникальные значения столбца p.gender: ['Женщины', 'Мужчины']
    * `value` (INTEGER): Численность населения. (Используй `p.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `p.municipal_district_name`)
3.  **migration** (алиас `mig`):
    * `territory_id` (STRING): (Используй `mig.territory_id`)
    * `year` (INTEGER): Данные за 2023 гг. (Используй `mig.year`)
    * `age` (STRING): (Используй `mig.age`). Все значения столбца mig.age это диапазоны, например: '19-22, '20-24', '75-79'
    * `gender` (STRING): (Используй `mig.gender`). Все уникальные значения mig.gender: ['Женщины', 'Мужчины']
    * `value` (INTEGER): Число мигрантов. (Используй `mig.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `mig.municipal_district_name`)
4.  **salary** (алиас `s`):
    * `territory_id` (STRING): (Используй `s.territory_id`)
    * `year` (INTEGER): С 1 кв. 2023 г. по 4 кв. 2024 г. (Используй `s.year`)
    * `period` (STRING): Квартал (например, '1 квартал', '2 квартал', '3 квартал', '4 квартал'). (Используй `s.period`). Все уникальные значения столбца s.period: ['январь-декабрь', 'январь-июнь', 'январь-март', 'январь-сентябрь']
    * `okved_name` (STRING): Наименование вида экономической деятельности (ОКВЭД). (Используй `s.okved_name`). Все уникальные значения столбца s.okved_name: ['Административная деятельность', 'Водоснабжение', 'Все отрасли', 'Гос. управление и военн. безопасность', 'Гостиницы и общепит', 'Деятельность экстер. организаций', 'Добыча полезных ископаемых', 'Здравоохранение', 'ИТ и связь', 'Научная и проф. деятельность', 'Обрабатывающие производства', 'Образование', 'Операции с недвижимостью', 'Прочие услуги', 'Сельское хозяйство', 'Спорт и досуг', 'Строительство', 'Торговля', 'Транспортировка и хранение', 'Услуги ЖКХ', 'Финансы и страхование']
    * `okved_letter` (STRING): Буквенное обозначение ОКВЭД. (Используй `s.okved_letter`). Все уникальные значения столбца s.okved_letter: ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'U']
    * `value` (FLOAT): Среднемесячная заработная плата. ВАЖНО: это и есть значение зарплаты, используй `s.value` для агрегаций типа AVG. (Используй `s.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `s.municipal_district_name`)
5.  **connections** (алиас `c`):
    * `territory_id_x` (STRING): ID МО отправления. (Используй `c.territory_id_x`)
    * `territory_id_y` (STRING): ID МО прибытия. (Используй `c.territory_id_y`)
    * `distance` (FLOAT): Расстояние в км. (Используй `c.distance`)

ПРИМЕЧАНИЕ ДЛЯ ГЕНЕРАЦИИ SQL: Всегда включай `municipal_district_name` в SELECT, если требуются названия территорий. Для строковых литералов в SQL используй одинарные кавычки. Убедись, что любая колонка, используемая в WHERE, SELECT, GROUP BY или ORDER BY, доступна из таблиц в FROM/JOIN и правильно квалифицирована алиасом.
"""


# ==============================================================================
# АГЕНТ 0: КООРДИНАТОР
# ==============================================================================

def agent0_coordinator(user_prompt):
    """Агент-координатор: определяет необходимость анализа аномалий и трендов"""
    system_instruction = (
        "Ты — ИИ-координатор запросов. Определи тип анализа, необходимый для запроса пользователя. "
        "Если запрос содержит слова или фразы, связанные с поиском аномалий, выбросов, необычных паттернов, "
        "резких изменений, отклонений от нормы, экстремальных значений, подозрительных данных, "
        "верни JSON: {\"needs_analysis\": \"anomaly\", \"keywords\": [\"найденные ключевые слова\"]}. "
        "Если запрос содержит слова или фразы, связанные с трендами, изменениями во времени, динамикой, "
        "ростом/падением, поворотными точками, переломами тренда, верни JSON: {\"needs_analysis\": \"trend\", \"keywords\": [\"найденные ключевые слова\"]}. "
        "Если запрос содержит оба типа анализа, верни JSON: {\"needs_analysis\": \"both\", \"keywords\": [\"ключевые слова аномалий\", \"ключевые слова трендов\"]}. "
        "Иначе верни JSON: {\"needs_analysis\": \"standard\", \"keywords\": []}. "
        "Ответ должен быть ТОЛЬКО JSON объектом без дополнительного текста."
    )
    response = call_giga_api_wrapper(user_prompt, system_instruction)

    try:
        result = json.loads(response)
        # Для обратной совместимости с существующим кодом
        if 'needs_analysis' in result:
            result['needs_anomaly_detection'] = result['needs_analysis'] in ['anomaly', 'both']
            result['needs_trend_analysis'] = result['needs_analysis'] in ['trend', 'both']
            result['analysis_type'] = result['needs_analysis']  # дублируем для совместимости
        return result
    except json.JSONDecodeError:
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass

        print(f"[COORDINATOR ERROR] Не удалось распарсить ответ: {response}")
        return {
            "needs_analysis": "standard",
            "needs_anomaly_detection": False,
            "needs_trend_analysis": False,
            "analysis_type": "standard",
            "keywords": []
        }


# ==============================================================================
# АГЕНТ 1: ПЕРЕФОРМУЛИРОВЩИК
# ==============================================================================

def agent1_rephraser(user_prompt):
    """Переформулирует запрос пользователя в формальный вопрос"""
    system_instruction = (
        "Ты — ИИ-ассистент. Переформулируй запрос пользователя в формальный, четкий вопрос для анализа данных на русском языке. "
        f"ВАЖНО: Если запрос касается аномалий но не указывает таблицу, предложи конкретную таблицу из доступных. "
        f"Доступные таблицы: population (население), salary (зарплаты), migration (миграция), "
        f"market_access (доступность рынков), connections (связи). "
        f"Вывод должен содержать ТОЛЬКО переформулированный вопрос.\n\n{TABLE_CONTEXT}"
    )
    return call_giga_api_wrapper(user_prompt, system_instruction)


# ==============================================================================
# АГЕНТ 2: ПЛАНИРОВЩИК
# ==============================================================================

def agent2_planner(formal_prompt):
    """Создает пошаговый план анализа данных"""
    system_instruction = (
        "Ты — ИИ-планировщик. На основе формального запроса создай краткий, пронумерованный пошаговый план на русском языке. "
        "Используй TABLE_CONTEXT. Вывод должен содержать ТОЛЬКО план. НЕ ВКЛЮЧАЙ SQL."
        f"\n\n{TABLE_CONTEXT}"
    )
    return call_giga_api_wrapper(formal_prompt, system_instruction)


# ==============================================================================
# АГЕНТ 3: ГЕНЕРАТОР SQL (С ПОДДЕРЖКОЙ АНОМАЛИЙ)
# ==============================================================================

def agent3_sql_generator(plan):
    """Генерирует SQL-запрос на основе плана с учетом анализа аномалий"""
    system_instruction = (
        "Ты — эксперт по генерации SQL. На основе плана и TABLE_CONTEXT сгенерируй SQL-запрос. "
        "ВАЖНО: ВСЕГДА используй алиасы для таблиц (например, `p` для `population`, `ma` для `market_access`, `s` для `salary`, `mig` для `migration`, `c` для `connections`) и ВСЕГДА квалифицируй КАЖДОЕ имя столбца этим алиасом (например, `p.year`, `md.territory_id`, `s.value` для зарплаты). Это КРИТИЧЕСКИ важно для избежания ошибок неоднозначности, особенно для `territory_id` и `year`. "

        "КРИТИЧЕСКИ ВАЖНО - ПРАВИЛА ДЛЯ ПОИСКА ПО НАЗВАНИЯМ МУНИЦИПАЛИТЕТОВ: "
        "Каждая таблица содержит столбцы `territory_id` И `municipal_district_name`. "
        "НИКОГДА НЕ используй точное сравнение (=) для поиска по названиям муниципалитетов! "
        "ВСЕГДА используй ТОЛЬКО нечеткий поиск через UPPER() и LIKE с процентами. "
        "Когда в запросе упоминается любое название места (Майкоп, Ломоносовском, майкопа, округа майкопа и т.д.), "
        "ты ОБЯЗАТЕЛЬНО должен: "
        "1. Извлечь ключевое слово из названия (например, из 'Ломоносовском' извлечь 'ЛОМОНОСОВ') "
        "2. Использовать ТОЛЬКО такой паттерн: WHERE UPPER(table_alias.municipal_district_name) LIKE '%КЛЮЧЕВОЕ_СЛОВО%' "
        "ЗАПРЕЩЕННЫЕ примеры: "
        "- WHERE p.municipal_district_name = 'Майкоп' "
        "- WHERE p.municipal_district_name = 'Ломоносовский' "
        "ПРАВИЛЬНЫЕ примеры: "
        "- WHERE UPPER(p.municipal_district_name) LIKE '%МАЙКОП%' "
        "- WHERE UPPER(p.municipal_district_name) LIKE '%ЛОМОНОСОВ%' "
        "Это найдет и 'Ломоносовский', и 'Ломоносовский муниципальный район', и любые другие варианты. "
        "Это обязательно для всех географических названий без исключений! "
        "ВСЕГДА включай столбец municipal_district_name в SELECT для отображения полного названия МО. "

        "ОСОБЫЕ ПРАВИЛА ДЛЯ АНОМАЛИЙ: "
        "Если план касается поиска аномалий, выбросов или необычных значений, "
        "НЕ включай в SQL логику фильтрации аномалий (WHERE value < lower_bound OR value > upper_bound). "
        "Вместо этого создай простой SELECT запрос, который возвращает ВСЕ данные из соответствующей таблицы "
        "с необходимыми JOIN для получения названий территорий. "
        "Агент аномалий выполнит анализ на полученных данных самостоятельно. "

        "Вывод должен быть ТОЛЬКО SQL-запрос в блоке ``````."
        f"\n\n{TABLE_CONTEXT}"
    )

    sql_query_block = call_giga_api_wrapper(plan, system_instruction)
    if sql_query_block and "```" in sql_query_block:
        try:
            return sql_query_block.split("```sql")[1].split("```")[0]
        except IndexError:
            return sql_query_block
    elif sql_query_block and sql_query_block.strip().upper().startswith("SELECT"):
        return sql_query_block.strip()
    return sql_query_block


# ==============================================================================
# АГЕНТ 4: ВАЛИДАТОР SQL
# ==============================================================================

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
        "5. Для таблицы 'salary' (алиас 's'), агрегации должны применяться к 's.value'.\n"
        "6. Для строковых литералов в SQL должны использоваться одинарные кавычки.\n"
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


# ==============================================================================
# АГЕНТ 5: ИСПРАВИТЕЛЬ SQL
# ==============================================================================

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
        "Вывод должен быть ТОЛЬКО исправленным SQL-запросом в блоке ```sql ... ```."
        f"\n\nПервоначальный план:\n{plan}\n\nСообщение об ошибке:\n{error_message}\n\n{TABLE_CONTEXT}"
    )
    prompt_for_llm = f"Неудавшийся SQL-запрос:\n```sql\n{failed_sql_query}\n```"
    corrected_sql_block = call_giga_api_wrapper(prompt_for_llm, system_instruction)

    if corrected_sql_block and "```sql" in corrected_sql_block:
        try:
            corrected_sql = corrected_sql_block.split("``````", 1)[0].strip()
            return corrected_sql if corrected_sql else None  # Возвращаем None если блок пустой
        except IndexError:
            return None
    return None


# ==============================================================================
# АГЕНТ АНОМАЛИЙ: ОБНАРУЖЕНИЕ И АНАЛИЗ
# ==============================================================================

def agent_anomaly_detector(sql_result_df, user_prompt):
    """Агент обнаружения аномалий в данных с исправленной JSON сериализацией"""
    if sql_result_df is None or sql_result_df.empty:
        return {"anomalies_found": False, "message": "Нет данных для анализа аномалий"}

    anomalies = []
    numeric_columns = sql_result_df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        return {"anomalies_found": False, "message": "В данных нет числовых столбцов для анализа"}

    # Для данных миграции применяем групповой анализ
    if 'territory_id' in sql_result_df.columns and 'value' in sql_result_df.columns:
        return analyze_migration_anomalies(sql_result_df, user_prompt)

    # Стандартный анализ для других типов данных
    for column in numeric_columns:
        column_data = sql_result_df[column].dropna()
        if len(column_data) < 4:
            continue

        # IQR метод
        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (sql_result_df[column] < lower_bound) | (sql_result_df[column] > upper_bound)
        outliers = sql_result_df[outliers_mask]

        if not outliers.empty:
            anomaly_info = {
                "column": column,
                "anomaly_count": len(outliers),
                "total_count": len(column_data),
                "anomaly_percentage": round((len(outliers) / len(column_data)) * 100, 2),
                "outlier_values": [float(x) for x in outliers[column].tolist()[:10]],  # Конвертируем в обычные float
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},  # Конвертируем в float
                "statistics": {
                    "mean": float(column_data.mean()),
                    "median": float(column_data.median()),
                    "std": float(column_data.std())
                }
            }
            anomalies.append(anomaly_info)

    if anomalies:
        try:
            anomaly_description = generate_anomaly_description(anomalies, user_prompt)
        except Exception as e:
            anomaly_description = f"Обнаружены аномалии в {len(anomalies)} столбцах"

        return {
            "anomalies_found": True,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "description": anomaly_description
        }
    else:
        return {"anomalies_found": False, "message": "Аномалии в данных не обнаружены"}


def analyze_migration_anomalies(df, user_prompt):
    """Специализированный анализ аномалий для миграционных данных"""
    anomalies = []

    # Группируем по территориям и анализируем
    for territory_id in df['territory_id'].unique():
        territory_data = df[df['territory_id'] == territory_id]['value']

        if len(territory_data) < 3:
            continue

        # Z-score метод для каждой территории
        mean_val = territory_data.mean()
        std_val = territory_data.std()

        if std_val == 0:  # Нет вариации
            continue

        z_scores = np.abs((territory_data - mean_val) / std_val)
        outlier_threshold = 2.5  # Более мягкий порог для миграционных данных

        outliers = territory_data[z_scores > outlier_threshold]

        if not outliers.empty:
            # ИСПРАВЛЕНИЕ: Правильное использование iloc с квадратными скобками
            if 'municipal_district_name' in df.columns:
                territory_rows = df[df['territory_id'] == territory_id]
                if not territory_rows.empty:
                    territory_name = territory_rows['municipal_district_name'].iloc[0]  # ИСПРАВЛЕНО!
                else:
                    territory_name = str(territory_id)
            else:
                territory_name = str(territory_id)

            anomaly_info = {
                "territory": territory_name,
                "territory_id": str(territory_id),  # Конвертируем в строку
                "anomaly_count": len(outliers),
                "total_count": len(territory_data),
                "anomaly_percentage": round((len(outliers) / len(territory_data)) * 100, 2),
                "outlier_values": [float(x) for x in outliers.tolist()],  # Конвертируем в float
                "statistics": {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "max_z_score": float(z_scores.max())
                }
            }
            anomalies.append(anomaly_info)

    if anomalies:
        description = f"Обнаружены миграционные аномалии в {len(anomalies)} территориях. Анализ выявил необычные миграционные потоки, которые значительно отклоняются от нормальных паттернов для данных территорий."

        return {
            "anomalies_found": True,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "description": description,
            "analysis_method": "Z-score по территориям"
        }

    return {
        "anomalies_found": False,
        "message": "Аномалии в миграционных данных не обнаружены",
        "analyzed_territories": len(df['territory_id'].unique())
    }


def generate_anomaly_description(anomalies, user_prompt):
    """Генерирует человекочитаемое описание найденных аномалий"""
    system_instruction = (
        "Ты — эксперт по анализу данных. На основе найденных аномалий создай краткое, "
        "понятное описание на русском языке. Объясни что обнаружено, в каких столбцах, "
        "и что это может означать. Будь конкретным и полезным."
    )

    anomaly_summary = "Найденные аномалии:\n"
    for anomaly in anomalies:
        anomaly_summary += f"- Столбец '{anomaly['column']}': {anomaly['anomaly_count']} аномальных значений ({anomaly['anomaly_percentage']}%)\n"

    prompt = f"Исходный запрос: {user_prompt}\n\n{anomaly_summary}"

    try:
        return call_giga_api_wrapper(prompt, system_instruction)
    except Exception as e:
        return f"Обнаружены аномалии, но не удалось сгенерировать описание: {str(e)}"


# ==============================================================================
# АГЕНТ 6: ГЕНЕРАТОР ОТВЕТОВ (ОПЦИОНАЛЬНЫЙ)
# ==============================================================================

def agent4_answer_generator(sql_result_df, initial_user_prompt, anomaly_results=None, trend_results=None):
    """Генерирует финальный ответ на основе данных, аномалий и трендов"""
    if sql_result_df is None:
        return "К сожалению, не удалось получить данные из-за ошибки SQL."
    if sql_result_df.empty:
        return "По вашему запросу данные не найдены."

    # Подготовка данных для LLM
    data_as_string = sql_result_df.to_markdown(index=False) if not sql_result_df.empty else "Данные отсутствуют."

    # Подготовка информации об аномалиях и трендах
    analysis_info = ""

    if anomaly_results and anomaly_results.get("anomalies_found", False):
        analysis_info += f"\n\nОбнаруженные аномалии:\n{anomaly_results.get('description', '')}"

    if trend_results and trend_results.get("trends_found", False):
        analysis_info += f"\n\nАнализ трендов:\n{trend_results.get('description', '')}"

    system_instruction = (
        "Ты — дружелюбный ИИ-ассистент, который объясняет результаты анализа данных простым и понятным языком. "
        "Тебе дан исходный вопрос пользователя, данные (результат SQL в Markdown) и дополнительная информация "
        "об аномалиях и трендах (если есть). Сформулируй краткий, приятный и легко читаемый ответ на русском языке. "
        "Избегай длинных и официальных названий, используй сокращённые или привычные формы, если это возможно. "
        "Сделай текст живым и дружелюбным, чтобы пользователю было приятно читать. "
        "Если видишь NaN, укажи, что данные отсутствуют. "
        "Сначала кратко ответь на основной вопрос, затем приведи ключевые выводы из анализа."
    )

    prompt_for_llm = (
        f"Исходный вопрос пользователя: \"{initial_user_prompt}\"\n\n"
        f"Данные для ответа (Markdown):\n{data_as_string}\n\n"
        f"Дополнительная информация:{analysis_info}\n\nОтвет:"
    )

    return call_giga_api_wrapper(prompt_for_llm, system_instruction)


# ==============================================================================
# АГЕНТ 7: АНАЛИЗ ТРЕНДОВ И ПОВОРОТНЫХ ТОЧЕК
# ==============================================================================
def generate_trend_description(trend_results, user_prompt):
    """Генерирует человекочитаемое описание найденных трендов"""
    system_instruction = (
        "Ты — эксперт по анализу временных рядов. На основе результатов анализа трендов создай "
        "понятное описание на русском языке. Опиши основные тренды, их направление и силу, "
        "а также обнаруженные поворотные точки. Будь конкретным и полезным."
    )

    trend_summary = "Результаты анализа трендов:\n"

    for trend in trend_results["trend_analysis"]:
        trend_summary += (
            f"- Столбец '{trend['column']}': {trend['trend_direction']} тренд ({trend['trend_strength']}), "
            f"R² = {trend['r_squared']:.2f}\n"
        )

    for point in trend_results["turning_points"]:
        trend_summary += (
            f"- Поворотная точка для '{point['column']}': {point['time_value']} "
            f"(значение: {point['value']:.2f})\n"
        )

    prompt = f"Исходный запрос: {user_prompt}\n\n{trend_summary}"

    try:
        return call_giga_api_wrapper(prompt, system_instruction)
    except Exception as e:
        return f"Обнаружены тренды, но не удалось сгенерировать описание: {str(e)}"


def agent_trend_analyzer(sql_result_df, user_prompt, kneed=None):
    """
    Анализирует временные ряды на наличие трендов и поворотных точек.
    Возвращает словарь с результатами анализа.
    """
    if sql_result_df is None or sql_result_df.empty:
        return {"trends_found": False, "message": "Нет данных для анализа трендов"}

    # Проверяем наличие временных данных
    time_columns = ['year', 'period']
    has_time_data = any(col in sql_result_df.columns for col in time_columns)

    if not has_time_data:
        return {"trends_found": False, "message": "Данные не содержат временной информации для анализа трендов"}

    results = {
        "trends_found": False,
        "trend_analysis": [],
        "turning_points": [],
        "description": ""
    }

    # Анализ для числовых столбцов
    numeric_columns = sql_result_df.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        if column in time_columns:
            continue

        # Подготовка данных для анализа
        if 'year' in sql_result_df.columns:
            time_col = 'year'
            df_sorted = sql_result_df.sort_values('year')
        elif 'period' in sql_result_df.columns:
            time_col = 'period'
            df_sorted = sql_result_df.sort_values('period')
        else:
            continue

        values = df_sorted[column].values
        times = df_sorted[time_col].values

        if len(values) < 3:
            continue

        # Анализ тренда (линейная регрессия)
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(range(len(values)), values)

            trend_info = {
                "column": column,
                "time_column": time_col,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "trend_direction": "возрастающий" if slope > 0 else "убывающий",
                "trend_strength": "сильный" if abs(slope) > np.std(values) else "слабый"
            }

            results["trend_analysis"].append(trend_info)
            results["trends_found"] = True
        except Exception as e:
            print(f"Ошибка анализа тренда для столбца {column}: {str(e)}")

        # Поиск поворотных точек (метод кусочно-линейной аппроксимации)
        try:
            from kneed import KneeLocator
            kl = KneeLocator(range(len(values)), values, curve='convex' if slope > 0 else 'concave',
                             direction='increasing')

            if kl.knee is not None:
                turning_point = {
                    "column": column,
                    "time_column": time_col,
                    "time_value": times[kl.knee],
                    "value": float(values[kl.knee]),
                    "index": int(kl.knee),
                    "method": "Kneedle algorithm"
                }
                results["turning_points"].append(turning_point)
        except Exception as e:
            print(f"Ошибка поиска поворотных точек для столбца {column}: {str(e)}")

    # Генерация описания результатов
    if results["trends_found"]:
        try:
            results["description"] = generate_trend_description(results, user_prompt)
        except Exception as e:
            results["description"] = f"Обнаружены тренды в {len(results['trend_analysis'])} столбцах"

    return results
