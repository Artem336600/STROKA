"""
Модуль для улучшенного семантического поиска с использованием Mistral AI API
"""
import requests
import logging
import json
from typing import List, Dict, Any, Optional

# Константы для Mistral API
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "6trhjxfirpthKPGAwm9jjtgWlfVKOgfa"

logger = logging.getLogger("mistral_search")

class MistralSearchEnhancer:
    """Класс для улучшения поиска с использованием Mistral API"""
    
    def __init__(self, model: str = "mistral-small"):
        """
        Инициализация класса для работы с Mistral API
        
        Args:
            model (str): Название модели Mistral для использования
        """
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Анализирует поисковый запрос для понимания глубинного смысла и категоризации
        
        Args:
            query (str): Поисковый запрос пользователя
            
        Returns:
            Dict[str, Any]: Результат анализа запроса
        """
        logger.info(f"Анализ запроса через Mistral: '{query}'")
        
        try:
            messages = [
                {"role": "system", "content": "Ты - помощник для анализа поисковых запросов. Твоя задача - понять глубинный смысл запроса и извлечь ключевую информацию."},
                {"role": "user", "content": f"""
                Проанализируй поисковый запрос: "{query}"
                
                Ответь в формате JSON со следующими полями:
                - search_type: тип поиска (skills, person, experience, education, general, profession)
                - extracted_entities: ключевые сущности из запроса
                - expanded_query: расширенный запрос с синонимами и связанными терминами на русском языке
                - search_categories: список категорий поиска
                
                Правила для анализа:
                1. Слово "программист" относится к профессии в сфере разработки ПО, его синонимы: разработчик, developer, coder, software engineer
                2. Профессии четко разделяй по сферам - инженер-механик НЕ программист
                3. Используй только русский язык в expanded_query, не смешивай русский и английский
                4. Определяй предметную область запроса и фокусируйся на ней
                5. Если запрос про профессию, устанавливай search_type: profession и добавляй в search_categories "profession_search"
                
                Например:
                - "кто знает питон" - ищем человека с навыками Python (языка программирования)
                - "инженер с опытом" - ищем инженера с опытом работы в технической сфере
                - "опытный программист" - ищем разработчика/программиста с большим опытом в IT
                """}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                MISTRAL_API_URL,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                try:
                    analysis = json.loads(content)
                    logger.info("Анализ запроса успешно выполнен")
                    return analysis
                except json.JSONDecodeError:
                    logger.error(f"Ошибка при разборе JSON ответа: {content}")
                    return {"error": "Не удалось разобрать JSON ответа"}
            else:
                logger.error(f"Неожиданный формат ответа от Mistral API: {result}")
                return {"error": "Неожиданный формат ответа"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к Mistral API: {e}")
            return {"error": f"Ошибка соединения с API: {str(e)}"}
        except Exception as e:
            logger.error(f"Неожиданная ошибка при анализе запроса: {e}")
            return {"error": f"Ошибка: {str(e)}"}
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Переранжирует результаты поиска с использованием Mistral API
        
        Args:
            query (str): Поисковый запрос пользователя
            results (List[Dict[str, Any]]): Исходные результаты поиска
            top_n (int): Количество результатов для возврата
            
        Returns:
            List[Dict[str, Any]]: Переранжированные результаты
        """
        if not results:
            return []
            
        logger.info(f"Переранжирование результатов ({len(results)}) через Mistral")
        
        try:
            # Подготавливаем контент для ранжирования
            profiles_json = json.dumps([{
                "id": i,
                "name": r.get("name", "Без имени"),
                "content": r.get("content", "")[0:500]  # Ограничиваем размер для API
            } for i, r in enumerate(results)])
            
            messages = [
                {"role": "system", "content": "Ты - поисковая система для ранжирования профилей. Твоя задача - оценить релевантность каждого профиля к запросу."},
                {"role": "user", "content": f"""
                Запрос пользователя: "{query}"
                
                Профили для ранжирования:
                {profiles_json}
                
                Оцени релевантность каждого профиля к запросу. Верни список идентификаторов профилей, 
                отсортированный по убыванию релевантности в формате JSON:
                {{
                  "ranked_ids": [id1, id2, id3, ...]
                }}
                
                Правила ранжирования:
                1. Четко различай профессии: программист и инженер-механик - разные специальности
                2. Программист относится к IT-сфере (разработчики, девелоперы, software engineers)
                3. Если запрос ищет специалиста в определенной области, профиль должен содержать соответствующие навыки
                4. Имя не должно быть фактором, если явно не ищется человек с определенным именем
                5. Не ранжируй высоко профили, которые не соответствуют предметной области запроса
                6. "Знает Python" означает навык программирования, а не знание о змеях
                7. Учитывай уровень опыта, указанный в запросе (например, "опытный" или "начинающий")
                """}
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                MISTRAL_API_URL,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                try:
                    ranking = json.loads(content)
                    
                    # Проверяем разные варианты ключей для ранжирования
                    ranked_ids = None
                    if "ranked_ids" in ranking:
                        ranked_ids = ranking["ranked_ids"]
                    elif "ranked-ids" in ranking:
                        ranked_ids = ranking["ranked-ids"]
                    
                    if ranked_ids:
                        # Переупорядочиваем результаты
                        ranked_results = []
                        
                        for id in ranked_ids:
                            if id < len(results):
                                ranked_results.append(results[id])
                                
                        logger.info(f"Успешно переранжировано: {len(ranked_results)} результатов")
                        return ranked_results[:top_n]
                    else:
                        logger.error(f"Отсутствует поле ranked_ids в ответе: {content}")
                        return results[:top_n]
                except json.JSONDecodeError:
                    logger.error(f"Ошибка при разборе JSON ответа: {content}")
                    return results[:top_n]
            else:
                logger.error(f"Неожиданный формат ответа от Mistral API: {result}")
                return results[:top_n]
                
        except Exception as e:
            logger.error(f"Ошибка при переранжировании через Mistral: {e}")
            return results[:top_n] 