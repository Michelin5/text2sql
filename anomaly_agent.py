## anomaly_agent.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import warnings
from langchain_core.messages import HumanMessage

warnings.filterwarnings('ignore')

class AnomalyDetectorAgent:
    """Агент для обнаружения аномалий в данных"""
    
    def __init__(self, llm_client, verbose=True):
        self.llm_client = llm_client
        self.name = "AnomalyDetector"
        self.role = "Специалист по обнаружению аномалий"
        self.verbose = verbose
        
    def call_giga_api_wrapper(self, prompt_text, system_instruction):
        """Обертка для вызова GigaChat API"""
        if self.llm_client is None:
            return "[ОШИБКА API] Клиент GigaChat не инициализирован."
        
        try:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_text)]
            res = self.llm_client.invoke(messages)
            return res.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"Ошибка вызова GigaChat API в агенте аномалий: {e}")
            return f"[ОШИБКА API] {e}"
    
    def detect_anomalies(self, sql_result_df, user_prompt):
        """Основной метод обнаружения аномалий"""
        
        if sql_result_df is None or sql_result_df.empty:
            return {
                "anomalies_found": False,
                "message": "Нет данных для анализа аномалий",
                "details": {},
                "agent": self.name
            }
        
        if self.verbose:
            print(f"\n🔍 {self.name}: Анализирую {len(sql_result_df)} записей...")
        
        # Выбираем числовые столбцы
        numeric_columns = sql_result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return {
                "anomalies_found": False,
                "message": "Нет числовых столбцов для анализа аномалий",
                "details": {},
                "agent": self.name
            }
        
        if self.verbose:
            print(f"📊 Анализирую числовые столбцы: {numeric_columns}")
        
        anomaly_results = {}
        
        try:
            # Метод 1: Z-Score для каждого числового столбца
            for col in numeric_columns:
                if sql_result_df[col].isnull().all():
                    continue
                    
                z_scores = np.abs(stats.zscore(sql_result_df[col].dropna()))
                threshold = 3.0
                anomaly_indices = np.where(z_scores > threshold)
                
                anomaly_results[col] = {
                    "method": "Z-Score",
                    "threshold": threshold,
                    "anomaly_count": len(anomaly_indices),
                    "anomaly_percentage": round((len(anomaly_indices) / len(sql_result_df)) * 100, 2),
                    "max_z_score": float(np.max(z_scores)) if len(z_scores) > 0 else 0,
                    "anomaly_indices": anomaly_indices.tolist()[:5]  # Первые 5 индексов
                }
            
            # Метод 2: IQR для числовых столбцов
            for col in numeric_columns:
                if sql_result_df[col].isnull().all():
                    continue
                
                Q1 = sql_result_df[col].quantile(0.25)
                Q3 = sql_result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Избегаем деления на ноль
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    anomaly_mask = (sql_result_df[col] < lower_bound) | (sql_result_df[col] > upper_bound)
                    anomaly_count = anomaly_mask.sum()
                    
                    anomaly_results[f"{col}_IQR"] = {
                        "method": "IQR",
                        "Q1": float(Q1),
                        "Q3": float(Q3),
                        "IQR": float(IQR),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "anomaly_count": int(anomaly_count),
                        "anomaly_percentage": round((anomaly_count / len(sql_result_df)) * 100, 2)
                    }
            
            # Метод 3: Isolation Forest (если достаточно данных)
            if len(sql_result_df) >= 20 and len(numeric_columns) >= 1:
                try:
                    df_clean = sql_result_df[numeric_columns].dropna()
                    if len(df_clean) >= 20:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(df_clean)
                        
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        predictions = iso_forest.fit_predict(data_scaled)
                        
                        anomaly_count = (predictions == -1).sum()
                        
                        anomaly_results["isolation_forest"] = {
                            "method": "Isolation Forest",
                            "contamination": 0.1,
                            "anomaly_count": int(anomaly_count),
                            "anomaly_percentage": round((anomaly_count / len(df_clean)) * 100, 2),
                            "analyzed_records": len(df_clean)
                        }
                except Exception as e:
                    if self.verbose:
                        print(f"Ошибка Isolation Forest: {e}")
            
            # Генерируем интерпретацию через LLM
            interpretation = self._generate_interpretation(anomaly_results, sql_result_df, user_prompt)
            
            # Определяем, найдены ли значимые аномалии
            significant_anomalies = any(
                result.get("anomaly_percentage", 0) > 5 
                for result in anomaly_results.values() 
                if isinstance(result, dict) and "anomaly_percentage" in result
            )
            
            return {
                "anomalies_found": significant_anomalies,
                "message": interpretation,
                "details": anomaly_results,
                "analyzed_columns": numeric_columns,
                "total_records": len(sql_result_df),
                "agent": self.name
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Ошибка в агенте анализа аномалий: {e}")
            return {
                "anomalies_found": False,
                "message": f"Ошибка при анализе аномалий: {str(e)}",
                "details": {},
                "agent": self.name
            }
    
    def _generate_interpretation(self, anomaly_results, data, user_prompt):
        """Генерация интерпретации результатов анализа аномалий"""
        
        # Упрощаем результаты для промпта
        simplified_results = {}
        for key, value in anomaly_results.items():
            if isinstance(value, dict) and "anomaly_percentage" in value:
                simplified_results[key] = {
                    "method": value.get("method", key),
                    "anomaly_count": value["anomaly_count"],
                    "anomaly_percentage": value["anomaly_percentage"]
                }
        
        prompt = f"""
        Проанализируй результаты обнаружения аномалий в данных и дай краткое объяснение на русском языке.
        
        Исходный запрос пользователя: {user_prompt}
        Общее количество записей: {len(data)}
        
        Результаты анализа аномалий:
        {json.dumps(simplified_results, ensure_ascii=False, indent=2)}
        
        Создай краткий отчет (максимум 150 слов), включающий:
        1. Общую оценку качества данных
        2. Найденные аномалии (если есть)
        3. Возможные причины аномалий
        4. Краткие рекомендации
        
        Отвечай только на русском языке, без технических деталей.
        """
        
        try:
            return self.call_giga_api_wrapper(prompt, "Ты - эксперт по анализу данных, специализирующийся на обнаружении аномалий.")
        except Exception as e:
            if self.verbose:
                print(f"Ошибка при генерации интерпретации аномалий: {e}")
            
            # Простая интерпретация без LLM
            total_anomalies = sum(
                result.get("anomaly_count", 0) 
                for result in simplified_results.values()
            )
            
            if total_anomalies > 0:
                return f"Обнаружено {total_anomalies} потенциальных аномалий в данных. Рекомендуется дополнительная проверка выявленных значений."
            else:
                return "Значительных аномалий в данных не обнаружено. Качество данных соответствует ожидаемым параметрам."

# Функции для определения необходимости анализа аномалий
def should_analyze_anomalies_by_data(sql_results_df, sql_query, config):
    """Определяет необходимость анализа аномалий на основе характеристик данных"""
    
    if sql_results_df is None or sql_results_df.empty:
        print("❌ Нет данных для анализа")
        return False
    
    criteria = {
        'has_numeric_data': len(sql_results_df.select_dtypes(include=[np.number]).columns) >= config["min_numeric_columns"],
        'sufficient_data': len(sql_results_df) >= config["min_records"],
        'multiple_regions': (
            'territory_id' in sql_results_df.columns and 
            sql_results_df['territory_id'].nunique() >= config["min_regions"]
        ) if config["min_regions"] > 0 else False,
        'salary_data': (
            config["enable_for_salary"] and (
                'salary' in sql_query.lower() or 
                any('value' in col.lower() for col in sql_results_df.columns) or
                's.value' in sql_query.lower()
            )
        ),
        'aggregated_data': (
            config["enable_for_aggregations"] and 
            any(keyword in sql_query.upper() for keyword in ['AVG', 'SUM', 'COUNT', 'MAX', 'MIN'])
        ),
        'time_series': (
            config["enable_for_time_series"] and 
            any(col in sql_results_df.columns for col in ['year', 'period', 'date'])
        )
    }
    
    print(f"📋 Подробные критерии анализа данных:")
    for key, value in criteria.items():
        print(f"   {key}: {value}")
    
    criteria_met = sum(criteria.values())
    threshold = config.get("auto_threshold", 2)
    
    print(f"📊 Критериев выполнено: {criteria_met} из {threshold} требуемых")
    
    return criteria_met >= threshold
