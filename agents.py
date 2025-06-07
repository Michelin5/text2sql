import json
import numpy as np
import pandas as pd
from giga_wrapper import call_giga_api_wrapper
from rag_handler import query_rag_db

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
    * `age` (STRING): Возрастная группа (например, 'Все возраста' для общего населения). (Используй `p.age`). Все значения столбца p.age: ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '65+', '66', '67', '68', '69', '7', '70+', '8', '9', 'Всего', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80+']
    * `gender` (STRING): Пол. (Используй `p.gender`). Все уникальные значения столбца p.gender: ['Женщины', 'Мужчины']
    * `value` (INTEGER): Численность населения. (Используй `p.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `p.municipal_district_name`)
3.  **migration** (алиас `mig`):
    * `territory_id` (STRING): (Используй `mig.territory_id`)
    * `year` (INTEGER): Данные за 2023 гг. (Используй `mig.year`)
    * `age` (STRING): (Используй `mig.age`). Все значения столбца mig.age это диапазоны, например: '19-22, '20-24', '75-79'
    * `gender` (STRING): (Используй `mig.gender`). Все уникальные значения mig.gender: ['Женщины', 'Мужчины']
    * `value` (INTEGER): Число мигрантов (если значение отрицательное - отток людей, если положительное - приток людей. (Используй `mig.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `mig.municipal_district_name`)
4.  **salary** (алиас `s`):
    * `territory_id` (STRING): (Используй `s.territory_id`)
    * `year` (INTEGER): С 1 кв. 2023 г. по 4 кв. 2024 г. (Используй `s.year`)
    * `period` (STRING): Квартал (например, '1 квартал', '2 квартал', '3 квартал', '4 квартал'). (Используй `s.period`). Все уникальные значения столбца s.period: ['январь-декабрь', 'январь-июнь', 'январь-март', 'январь-сентябрь']
    * `okved_name` (STRING): Наименование вида экономической деятельности (ОКВЭД) (отрасли). (Используй `s.okved_name`). Все уникальные значения столбца s.okved_name: ['Административная деятельность', 'Водоснабжение', 'Все отрасли', 'Гос. управление и военн. безопасность', 'Гостиницы и общепит', 'Деятельность экстер. организаций', 'Добыча полезных ископаемых', 'Здравоохранение', 'ИТ и связь', 'Научная и проф. деятельность', 'Обрабатывающие производства', 'Образование', 'Операции с недвижимостью', 'Прочие услуги', 'Сельское хозяйство', 'Спорт и досуг', 'Строительство', 'Торговля', 'Транспортировка и хранение', 'Услуги ЖКХ', 'Финансы и страхование']
    * `okved_letter` (STRING): Буквенное обозначение ОКВЭД. (Используй `s.okved_letter`). Все уникальные значения столбца s.okved_letter: ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'U']
    * `value` (FLOAT): Среднемесячная заработная плата. ВАЖНО: это и есть значение зарплаты В РУБЛЯХ, используй `s.value` для агрегаций типа AVG. (Используй `s.value`)
    * `municipal_district_name` (STRING): Наименование МО. (Используй `s.municipal_district_name`)
5.  **connections** (алиас `c`):
    * `territory_id_x` (STRING): ID МО отправления. (Используй `c.territory_id_x`)
    * `territory_id_y` (STRING): ID МО прибытия. (Используй `c.territory_id_y`)
    * `distance` (FLOAT): Расстояние в км. (Используй `c.distance`)

ПРИМЕЧАНИЕ ДЛЯ ГЕНЕРАЦИИ SQL: Всегда включай `municipal_district_name` в SELECT, если требуются названия территорий. Для строковых литералов в SQL используй одинарные кавычки. Убедись, что любая колонка, используемая в WHERE, SELECT, GROUP BY или ORDER BY, доступна из таблиц в FROM/JOIN и правильно квалифицирована алиасом.
"""

# ==============================================================================
# АГЕНТ RAG: ПОИСК В БАЗЕ УСПЕШНЫХ ЗАПРОСОВ
# ==============================================================================
def agent_rag_retriever(user_prompt: str):
    """
    Агент для поиска похожих запросов в базе данных RAG.
    Возвращает результат от query_rag_db.
    """
    print(f"[AGENT RAG RETRIEVER] Querying RAG for prompt: {user_prompt[:100]}...")
    rag_result = query_rag_db(user_prompt)
    print(f"[AGENT RAG RETRIEVER] RAG Result: {rag_result.get('status')}")
    if rag_result.get('status') == 'similar' or rag_result.get('status') == 'exact':
        print(f"  [AGENT RAG RETRIEVER] Found matching data: {str(rag_result.get('data', {}))[:200]}...") # Log snippet of data
    return rag_result

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

def agent1_rephraser(user_prompt: str, rag_context: dict = None):
    """
    Переформулирует запрос пользователя в формальный вопрос с консервативным использованием RAG
    и поддержкой 90% совпадений на основе исследований multi-agent query reformulation
    """
    
    # ПРОВЕРКА 90% СОВПАДЕНИЯ С ПРОШЛЫМ ЗАПРОСОМ
    if rag_context and isinstance(rag_context, dict) and rag_context.get('user_prompt'):
        similarity_score = calculate_query_similarity(user_prompt, rag_context.get('user_prompt', ''))
        
        print(f"[REPHRASER] Сходство запросов: {similarity_score:.3f}")
        
        # Если 90%+ совпадение - возвращаем прошлую формализацию
        if similarity_score >= 0.90:
            past_formal = rag_context.get('formal_prompt', '')
            if past_formal:
                print(f"[REPHRASER] 90%+ совпадение - используем прошлую формализацию")
                return past_formal
    
    # КОНСЕРВАТИВНОЕ RAG guidance на основе исследований semantic matching
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('formal_prompt'):
        # Проверяем семантическую близость для contextual enrichment
        current_keywords = set(_normalize_query_tokens(user_prompt.lower().split()))
        past_keywords = set(_normalize_query_tokens(rag_context.get('user_prompt', '').lower().split()))
        
        # Semantic matching threshold на основе исследований
        keyword_overlap = len(current_keywords.intersection(past_keywords)) / max(len(current_keywords), 1)
        
        # СТРОГИЙ порог для RAG контекста (70%+ совпадение по ключевым словам)
        if keyword_overlap > 0.7:
            rag_guidance = (
                "ВАЖНО: Используй предоставленный контекст ТОЛЬКО как стилистическое руководство. "
                "Твоя основная задача - формализовать ТЕКУЩИЙ запрос пользователя. "
                "НЕ копируй прошлый formal_prompt, а создай новый на основе ТЕКУЩЕГО запроса.\n"
                f"Стилистический пример (для понимания подхода):\n"
                f"  Прошлый запрос: {rag_context.get('user_prompt', 'N/A')}\n"
                f"  Как был формализован: {rag_context.get('formal_prompt', 'N/A')}\n"
                f"Используй этот СТИЛЬ, но формализуй ТЕКУЩИЙ запрос: '{user_prompt}'\n\n"
            )
        else:
            # Query simplification approach - игнорируем несмежный RAG
            rag_guidance = (
                "Предоставленный контекст из прошлого запроса не релевантен текущему. "
                "Сосредоточься ТОЛЬКО на формализации текущего запроса пользователя.\n\n"
            )

    # SYSTEM INSTRUCTION на основе multi-agent query reformulation research
    system_instruction = f"""
Ты — эксперт по переформулировке запросов на основе исследований query reformulation и semantic matching.

{rag_guidance}

КРИТИЧЕСКИ ВАЖНО (основано на исследованиях query reformulation):
1. ВСЕГДА формализуй ТЕКУЩИЙ запрос: "{user_prompt}"
2. Применяй методы contextual enrichment и query expansion
3. Сохраняй ВСЕ ключевые намерения пользователя (intent preservation)
4. Используй semantic matching для улучшения точности
5. Делай минимальные изменения - только для ясности (query simplification)

МЕТОДЫ ФОРМАЛИЗАЦИИ (на основе исследований):
- Contextual Enrichment: добавь контекст если неясно ("данные" → "данные о населении")
- Query Expansion: уточни неопределенные термины ("топ 5" → "топ 5 регионов по показателю")
- Temporal Context: сохрани временные рамки если указаны
- Entity Recognition: укажи подходящую таблицу если понятно из контекста

ДОСТУПНЫЕ ТАБЛИЦЫ (для entity recognition):
- population: данные о населении (территория, год, пол, возраст, значение)
- salary: зарплаты (территория, год, экономическая деятельность, значение)
- migration: миграция (территория, год, пол, тип миграции, значение)
- market_access: доступность рынков (территория, значение)
- connections: связи (территория, год, значение)

ПРИМЕРЫ КОНСЕРВАТИВНОЙ ФОРМАЛИЗАЦИИ (semantic matching):
- "Найди аномалии" → "Какие аномальные значения присутствуют в данных о населении?"
- "Топ зарплат" → "Какие 5 регионов имеют самые высокие средние зарплаты?"
- "Тренды населения" → "Как изменялась численность населения по годам?"

ПРИНЦИПЫ (из исследований):
- Intent preservation: сохраняй намерение пользователя
- Semantic alignment: выравнивай с доменной терминологией  
- Contextual clarity: добавляй контекст только при необходимости
- Minimal modification: минимальные изменения для максимальной ясности

Ответь ТОЛЬКО формализованным вопросом без дополнительных пояснений.
"""

    try:
        from giga_wrapper import call_giga_api_wrapper
        
        # Iterative refinement approach - простой промпт для фокуса на текущем запросе
        prompt = f"""
Исходный запрос пользователя: "{user_prompt}"

Применяя методы semantic matching и contextual enrichment, сформулируй четкий формальный вопрос для анализа данных, точно сохраняя все намерения пользователя.
"""
        
        formal_prompt_result = call_giga_api_wrapper(prompt, system_instruction)
        
        # ВАЛИДАЦИЯ качества переформулировки (на основе исследований)
        if not _validate_reformulation_quality(user_prompt, formal_prompt_result):
            print(f"[REPHRASER WARNING] Переформулировка может быть неточной. Используем консервативный fallback.")
            return _conservative_semantic_fallback(user_prompt)
        
        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: нет ли подмены запроса
        if _is_query_substitution(user_prompt, formal_prompt_result, rag_context):
            print(f"[REPHRASER WARNING] Обнаружена подмена запроса. Используем консервативный подход.")
            return _conservative_semantic_fallback(user_prompt)
        
        return formal_prompt_result.strip()
    
    except Exception as e:
        print(f"[REPHRASER ERROR] Ошибка переформулировки: {e}")
        return _conservative_semantic_fallback(user_prompt)

def calculate_query_similarity(query1: str, query2: str) -> float:
    """
    Вычисляет семантическую близость запросов на основе исследований similarity search
    """
    if not query1 or not query2:
        return 0.0
    
    query1_lower = query1.lower().strip()
    query2_lower = query2.lower().strip()
    
    # Точное совпадение
    if query1_lower == query2_lower:
        return 1.0
    
    # Semantic matching через токенизацию
    tokens1 = set(_normalize_query_tokens(query1_lower.split()))
    tokens2 = set(_normalize_query_tokens(query2_lower.split()))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Jaccard similarity (базовое пересечение)
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard_sim = len(intersection) / len(union) if union else 0.0
    
    # Weighted semantic matching (важные слова имеют больший вес)
    domain_keywords = ['аномали', 'топ', 'анализ', 'данные', 'населен', 'зарплат', 'мигра', 'тренд', 'выброс', 'рейтинг']
    semantic_matches = sum(1 for word in domain_keywords if word in query1_lower and word in query2_lower)
    semantic_weight = semantic_matches * 0.15
    
    # Positional similarity (учет порядка слов)
    positional_sim = _calculate_positional_similarity(query1_lower.split(), query2_lower.split())
    
    # Length ratio penalty (штраф за разную длину)
    len_ratio = min(len(query1_lower), len(query2_lower)) / max(len(query1_lower), len(query2_lower))
    length_bonus = len_ratio * 0.1
    
    # Итоговый скор semantic matching
    final_similarity = jaccard_sim * 0.6 + semantic_weight + positional_sim * 0.2 + length_bonus
    
    return min(1.0, final_similarity)

def _normalize_query_tokens(tokens: list) -> list:
    """Нормализует токены для semantic matching"""
    stop_words = {'в', 'на', 'по', 'для', 'из', 'с', 'и', 'а', 'но', 'или', 'что', 'как', 'где', 'когда'}
    normalized = []
    
    for token in tokens:
        # Очистка от пунктуации
        clean_token = ''.join(char for char in token if char.isalnum())
        if clean_token and clean_token not in stop_words and len(clean_token) > 1:
            # Простой stemming (убираем окончания)
            if clean_token.endswith(('ах', 'ях', 'ами', 'ями')):
                clean_token = clean_token[:-3]
            elif clean_token.endswith(('ов', 'ев', 'ах', 'ям', 'ми')):
                clean_token = clean_token[:-2]
            elif clean_token.endswith(('а', 'я', 'е', 'и', 'о', 'у', 'ы')):
                clean_token = clean_token[:-1]
            
            normalized.append(clean_token)
    
    return normalized

def _calculate_positional_similarity(tokens1: list, tokens2: list) -> float:
    """Positional similarity для учета порядка слов"""
    if not tokens1 or not tokens2:
        return 0.0
    
    common_tokens = set(tokens1).intersection(set(tokens2))
    if not common_tokens:
        return 0.0
    
    position_matches = 0
    for token in common_tokens:
        try:
            pos1 = tokens1.index(token) / len(tokens1)
            pos2 = tokens2.index(token) / len(tokens2)
            # Близкие позиции дают бонус
            if abs(pos1 - pos2) < 0.3:
                position_matches += 1
        except (ValueError, ZeroDivisionError):
            continue
    
    return position_matches / len(common_tokens) if common_tokens else 0.0

def _validate_reformulation_quality(original_prompt: str, reformulated_prompt: str) -> bool:
    """Валидация качества переформулировки на основе исследований"""
    
    if not reformulated_prompt or len(reformulated_prompt.strip()) == 0:
        return False
    
    # Semantic preservation check
    original_words = set(_normalize_query_tokens(original_prompt.lower().split()))
    reformulated_words = set(_normalize_query_tokens(reformulated_prompt.lower().split()))
    
    # Должно быть хотя бы 40% общих семантических единиц
    common_words = original_words.intersection(reformulated_words)
    overlap_ratio = len(common_words) / max(len(original_words), 1)
    
    if overlap_ratio < 0.4:
        return False
    
    # Intent preservation check - ключевые индикаторы намерения
    intent_indicators = ['аномали', 'топ', 'тренд', 'анализ', 'зарплат', 'населен', 'мигра', 'выброс', 'рейтинг']
    original_intent = [ind for ind in intent_indicators if ind in original_prompt.lower()]
    reformulated_intent = [ind for ind in intent_indicators if ind in reformulated_prompt.lower()]
    
    # Если в оригинале были ключевые намерения, они должны сохраниться
    if original_intent and not any(intent in reformulated_intent for intent in original_intent):
        return False
    
    return True

def _is_query_substitution(original_prompt: str, reformulated_prompt: str, rag_context: dict = None) -> bool:
    """Проверяет, не произошла ли подмена запроса на прошлый из RAG"""
    
    if not rag_context or not rag_context.get('user_prompt'):
        return False
    
    past_prompt = rag_context.get('user_prompt', '')
    past_formal = rag_context.get('formal_prompt', '')
    
    # Проверяем, не слишком ли формализация похожа на прошлую
    if past_formal and reformulated_prompt:
        past_similarity = calculate_query_similarity(reformulated_prompt, past_formal)
        current_similarity = calculate_query_similarity(reformulated_prompt, original_prompt)
        
        # Если формализация больше похожа на прошлую, чем на текущую - подозрение на подмену
        if past_similarity > current_similarity + 0.3:
            return True
    
    return False

def _conservative_semantic_fallback(user_prompt: str) -> str:
    """Консервативная резервная формализация на основе semantic patterns"""
    
    prompt_lower = user_prompt.lower()
    
    # Semantic pattern matching для доменных задач
    if any(word in prompt_lower for word in ['аномали', 'выброс', 'необычн', 'странн']):
        if any(word in prompt_lower for word in ['населен', 'демограф', 'людей']):
            return f"Какие аномальные значения присутствуют в данных о населении?"
        elif any(word in prompt_lower for word in ['зарплат', 'заработ', 'доход']):
            return f"Какие аномальные значения присутствуют в данных о зарплатах?"
        elif any(word in prompt_lower for word in ['мигра', 'переезд']):
            return f"Какие аномальные значения присутствуют в миграционных данных?"
        else:
            return f"Какие аномальные значения присутствуют в данных?"
    
    elif any(word in prompt_lower for word in ['топ', 'лучш', 'высок', 'рейтинг', 'максимальн']):
        if any(word in prompt_lower for word in ['зарплат', 'заработ', 'доход']):
            return f"Какие регионы имеют самые высокие показатели по зарплатам?"
        elif any(word in prompt_lower for word in ['населен', 'демограф']):
            return f"Какие регионы имеют самые высокие показатели по населению?"
        else:
            return f"Какие регионы имеют самые высокие показатели?"
    
    elif any(word in prompt_lower for word in ['тренд', 'динамик', 'изменени', 'временн']):
        if any(word in prompt_lower for word in ['населен', 'демограф']):
            return f"Как изменялась численность населения по годам?"
        elif any(word in prompt_lower for word in ['зарплат', 'заработ']):
            return f"Как изменялись зарплаты по годам?"
        else:
            return f"Как изменялись показатели по годам?"
    
    else:
        # Минимальная contextual enrichment
        return f"Проанализируйте данные согласно запросу: {user_prompt}"



# ==============================================================================
# АГЕНТ 2: ПЛАНИРОВЩИК
# ==============================================================================

def agent2_planner(formal_prompt: str, rag_context: dict = None):
    """Создает пошаговый план анализа данных"""
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('plan'):
        rag_guidance = (
            "БАЗОВЫЙ ПЛАН ИЗ ПОХОЖЕГО ЗАПРОСА:\n"
            f"{rag_context.get('plan', 'N/A')}\n"
            f"РЕЗУЛЬТИРУЮЩИЙ SQL:\n{rag_context.get('sql_query', 'N/A')}\n\n"
            "АДАПТИРУЙ этот план под текущий запрос. Если запрос идентичен - используй план как есть. "
            "Если есть отличия - модифицируй соответствующие пункты.\n\n"
        )

    system_instruction = (
        "Ты создаешь технический план для SQL-генератора. "
        f"{rag_guidance}"
        "ТРЕБОВАНИЯ К ПЛАНУ:\n"
        "- Только пронумерованные пункты на русском языке\n"
        "- Каждый пункт = конкретное действие с данными\n"
        "- Используй точные названия таблиц и полей из TABLE_CONTEXT\n"
        "- Указывай конкретные условия фильтрации и группировки\n"
        "- Запрещено: SQL-код, абстрактные формулировки, лишние объяснения\n"
        "- Разрешено: 'выбрать поля X,Y из таблицы Z', 'отфильтровать по условию A=B', 'сгруппировать по полю C' и так далее\n\n"
        f"\\n\\n{TABLE_CONTEXT}"
    )
    return call_giga_api_wrapper(formal_prompt, system_instruction)


# ==============================================================================
# АГЕНТ 3: ГЕНЕРАТОР SQL (С ПОДДЕРЖКОЙ АНОМАЛИЙ)
# ==============================================================================

def agent3_sql_generator(plan: str, rag_context: dict = None):
    """Генерирует SQL-запрос на основе плана с учетом анализа аномалий"""
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('sql_query'):
        rag_guidance = (
            "Тебе также предоставлен контекст из очень похожего успешного прошлого запроса, включая его SQL-запрос. "
            "Если текущий план очень похож на тот, что привел к прошлому SQL (см. `rag_context.plan`), "
            "используй прошлый SQL-запрос (`rag_context.sql_query`) как очень сильную основу. "
            "Адаптируй его под текущий план, если есть небольшие отличия в именах столбцов, условиях фильтрации и т.д. "
            "Убедись, что новый SQL соответствует ТЕКУЩЕМУ плану. "
            "Если текущий план сильно отличается, сгенерируй SQL с нуля.\n"
            f"Контекст прошлого запроса (план):\n{rag_context.get('plan', 'N/A')}\n"
            f"Контекст прошлого запроса (SQL):\n```sql\n{rag_context.get('sql_query', 'N/A')}\n```\n\n"
        )
    
    system_instruction = (
        "Ты — эксперт по генерации SQL. На основе плана и TABLE_CONTEXT сгенерируй SQL-запрос. "
        f"{rag_guidance}"
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
    """Оптимизированный агент обнаружения аномалий - только топ-10 результатов"""
    if sql_result_df is None or sql_result_df.empty:
        return {"anomalies_found": False, "message": "Нет данных для анализа аномалий"}

    # ОГРАНИЧИВАЕМ АНАЛИЗ ПЕРВЫМИ 1000 ЗАПИСЯМИ ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ
    df_sample = sql_result_df.head(1000) if len(sql_result_df) > 1000 else sql_result_df
    
    anomalies = []
    numeric_columns = df_sample.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        return {"anomalies_found": False, "message": "В данных нет числовых столбцов для анализа"}

    # Для данных миграции
    if 'territory_id' in df_sample.columns and 'value' in df_sample.columns:
        return analyze_migration_anomalies_optimized(df_sample, user_prompt)

    # Стандартный анализ - ТОЛЬКО ПЕРВЫЙ ЧИСЛОВОЙ СТОЛБЕЦ
    for column in numeric_columns[:1]:  # Анализируем только первый столбец
        column_data = df_sample[column].dropna()
        if len(column_data) < 4:
            continue

        Q1 = column_data.quantile(0.25)
        Q3 = column_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (df_sample[column] < lower_bound) | (df_sample[column] > upper_bound)
        outliers = df_sample[outliers_mask]

        if not outliers.empty and len(outliers) >= 3:  # Минимум 3 аномалии для отчета
            # ОГРАНИЧИВАЕМ ДО 10 ПРИМЕРОВ
            top_outliers = outliers.head(10)
            
            anomaly_info = {
                "column": column,
                "anomaly_count": len(outliers),
                "total_count": len(column_data),
                "anomaly_percentage": round((len(outliers) / len(column_data)) * 100, 2),
                "outlier_values": [float(x) for x in top_outliers[column].tolist()],
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                "statistics": {
                    "mean": float(column_data.mean()),
                    "median": float(column_data.median()),
                    "std": float(column_data.std())
                }
            }
            anomalies.append(anomaly_info)

    if anomalies:
        return {
            "anomalies_found": True,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "description": f"Обнаружены аномалии в {len(anomalies)} показателях. Найдено {anomalies[0]['anomaly_count']} аномальных значений ({anomalies[0]['anomaly_percentage']}%)."
        }
    else:
        return {"anomalies_found": False, "message": "Аномалии в данных не обнаружены"}

def analyze_migration_anomalies_optimized(df, user_prompt):
    """Оптимизированный анализ миграционных аномалий - топ-10"""
    anomalies = []
    
    # Берем только топ-10 территорий по количеству данных
    territory_counts = df['territory_id'].value_counts().head(10)
    
    for territory_id in territory_counts.index:
        territory_data = df[df['territory_id'] == territory_id]['value']
        
        if len(territory_data) < 3:
            continue

        mean_val = territory_data.mean()
        std_val = territory_data.std()

        if std_val == 0:
            continue

        z_scores = np.abs((territory_data - mean_val) / std_val)
        outliers = territory_data[z_scores > 2.5]

        if not outliers.empty:
            territory_name = df[df['territory_id'] == territory_id]['municipal_district_name'].iloc[0] if 'municipal_district_name' in df.columns else str(territory_id)

            anomaly_info = {
                "territory": territory_name,
                "territory_id": str(territory_id),
                "anomaly_count": len(outliers),
                "total_count": len(territory_data),
                "anomaly_percentage": round((len(outliers) / len(territory_data)) * 100, 2),
                "outlier_values": [float(x) for x in outliers.head(5).tolist()],  # Только 5 примеров
                "statistics": {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "max_z_score": float(z_scores.max())
                }
            }
            anomalies.append(anomaly_info)

    if anomalies:
        # ОГРАНИЧИВАЕМ ДО 10 ТЕРРИТОРИЙ
        anomalies = anomalies[:10]
        
        description = f"Обнаружены миграционные аномалии в {len(anomalies)} территориях из топ-10 анализируемых."
        return {
            "anomalies_found": True,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "description": description,
            "analysis_method": "Z-score по территориям (топ-10)"
        }

    return {
        "anomalies_found": False,
        "message": "Аномалии в миграционных данных не обнаружены",
        "analyzed_territories": len(territory_counts)
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
# АГЕНТ 6: ГЕНЕРАТОР ОТВЕТА
# ==============================================================================

def agent4_answer_generator(sql_result_df, initial_user_prompt: str, query_plan: str, sql_query: str, 
                            anomaly_results: dict = None, trend_results: dict = None, rag_context: dict = None):
    """Генерирует текстовый ответ на основе результатов SQL и первоначального запроса."""
    rag_guidance = ""
    if rag_context and isinstance(rag_context, dict) and rag_context.get('final_answer'):
        rag_guidance = (
            "Тебе также предоставлен контекст из очень похожего успешного прошлого запроса, включая его финальный ответ. "
            "Используй стиль, тон и структуру прошлого ответа (`rag_context.final_answer`) как хороший пример. "
            "Однако, убедись, что ТЕКУЩИЙ ответ ТОЧНО отражает предоставленные РЕЗУЛЬТАТЫ SQL (`sql_result_df`), "
            "а не данные из прошлого ответа. Адаптируй формулировки под текущие данные.\n"
            f"Контекст прошлого запроса (финальный ответ):\n{rag_context.get('final_answer', 'N/A')}\n\n"
        )

    if sql_result_df is None:
        return "К сожалению, не удалось получить данные для вашего запроса."
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