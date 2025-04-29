"""
Модуль для поиска по профилям
"""
import re
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Импортируем класс MistralSearchEnhancer
from mistral_search import MistralSearchEnhancer

logger = logging.getLogger("search")

class Searcher:
    """Класс для поиска по профилям"""
    
    def __init__(
        self, 
        profiles: List[Dict[str, Any]], 
        embedding_model_name: str = "distiluse-base-multilingual-cased-v2",
        vector_weight: float = 0.7,
        default_limit: int = 5,
        use_mistral: bool = True,
        mistral_model: str = "mistral-small"
    ):
        self.profiles = profiles
        self.model_name = embedding_model_name
        self.vector_weight = vector_weight
        self.default_limit = default_limit
        self._embedder = None
        self.embeddings = None
        self.use_mistral = use_mistral
        
        # Инициализируем MistralSearchEnhancer если use_mistral=True
        self._mistral_enhancer = MistralSearchEnhancer(model=mistral_model) if use_mistral else None
        
        # Шаблоны для анализа запросов
        self.query_patterns = {
            "name_search": [
                re.compile(r'(?:найти|поиск|информация о|данные о|кто такой|кто такая)\s+([А-Я][а-я]+\s+[А-Я][а-я]+)', re.IGNORECASE),
                re.compile(r'(?:найти|поиск|информация о|данные о)\s+([А-Я][а-я]+)', re.IGNORECASE)
            ],
            "skills_search": [
                re.compile(r'(?:навыки|умения|знает|умеет|специалист\s+по)\s+([^?\.]+)', re.IGNORECASE)
            ],
            "experience_search": [
                re.compile(r'(?:опыт|стаж|работал)\s+(\d+)\s+лет', re.IGNORECASE),
                re.compile(r'опыт работы\s+([^?\.]+)', re.IGNORECASE)
            ],
            "education_search": [
                re.compile(r'(?:образование|учился|окончил|университет)\s+([^?\.]+)', re.IGNORECASE)
            ]
        }
        
        # Инициализация эмбеддингов
        self._create_embeddings()
    
    @property
    def embedder(self) -> SentenceTransformer:
        """Ленивая инициализация модели эмбеддингов"""
        if self._embedder is None:
            logger.info(f"Инициализация модели эмбеддингов: {self.model_name}")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def _create_embeddings(self):
        """Создает векторные представления для всех профилей"""
        if not self.profiles:
            return
            
        logger.info(f"Создание векторных представлений для {len(self.profiles)} профилей...")
        
        # Извлекаем тексты для эмбеддингов
        texts = [profile['content'] for profile in self.profiles]
        
        # Создаем эмбеддинги
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        logger.info("Создание векторных представлений завершено")
    
    def search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Выполняет поиск по профилям"""
        if not query.strip():
            logger.warning("Пустой поисковый запрос")
            return []
            
        if len(self.profiles) == 0 or self.embeddings is None:
            logger.warning("Профили не загружены")
            return []
            
        logger.info(f"Поиск: \"{query}\"")
        start_time = time.time()
        limit = limit or self.default_limit
        
        # Анализируем запрос с использованием Mistral, если доступен
        if self.use_mistral and self._mistral_enhancer:
            mistral_analysis = self._mistral_enhancer.analyze_query(query)
            logger.info(f"Анализ запроса через Mistral: {mistral_analysis}")
            
            # Проверяем на ошибки
            if "error" in mistral_analysis:
                logger.warning(f"Ошибка анализа через Mistral: {mistral_analysis['error']}")
                query_analysis = self.analyze_query(query)  # Используем стандартный анализ как запасной вариант
            else:
                # Используем информацию из Mistral для улучшения поиска
                query_analysis = {
                    'original_query': query,
                    'categories': mistral_analysis.get('search_categories', []),
                    'search_type': mistral_analysis.get('search_type', 'general'),
                    'entities': mistral_analysis.get('extracted_entities', [])
                }
                
                # Предварительная фильтрация по профессии
                if query_analysis['search_type'] == 'profession':
                    profession_keywords = []
                    for entity in query_analysis.get('entities', []):
                        if isinstance(entity, str) and len(entity) > 3:  # Минимальная длина для профессии
                            profession_keywords.append(entity.lower())
                    
                    if profession_keywords:
                        logger.info(f"Фильтрация по профессии: {profession_keywords}")
                        filtered_profiles = []
                        filtered_embeddings = []
                        
                        for i, profile in enumerate(self.profiles):
                            content_lower = profile['content'].lower()
                            
                            # Проверяем совпадение ключевых слов профессии
                            matches = False
                            for keyword in profession_keywords:
                                if keyword in content_lower:
                                    matches = True
                                    break
                            
                            if matches:
                                filtered_profiles.append(profile)
                                filtered_embeddings.append(self.embeddings[i])
                        
                        # Если найдены соответствующие профили, используем только их
                        if filtered_profiles:
                            logger.info(f"Найдено {len(filtered_profiles)} профилей, соответствующих профессии")
                            temp_profiles = self.profiles
                            temp_embeddings = self.embeddings
                            
                            # Временно заменяем списки профилей и эмбеддингов
                            self.profiles = filtered_profiles
                            self.embeddings = np.array(filtered_embeddings)
                            
                            # Получаем результаты поиска
                            results = self._perform_vector_search(query_analysis, query, limit)
                            
                            # Восстанавливаем исходные списки
                            self.profiles = temp_profiles
                            self.embeddings = temp_embeddings
                            
                            end_time = time.time()
                            logger.info(f"Поиск выполнен за {end_time - start_time:.3f}с. Найдено {len(results)} результатов")
                            return results
                
                # Проверяем наличие расширенного запроса
                if 'expanded_query' in mistral_analysis and mistral_analysis['expanded_query']:
                    # Для векторного поиска можем использовать расширенный запрос
                    expanded_query = mistral_analysis['expanded_query']
                    logger.info(f"Используем расширенный запрос: {expanded_query}")
                    query_for_vector = expanded_query
                else:
                    query_for_vector = query
        else:
            # Используем стандартный анализ, если Mistral недоступен
            query_analysis = self.analyze_query(query)
            query_for_vector = query
            
        # Прямой поиск по имени
        if 'direct_name' in query_analysis or ('search_type' in query_analysis and query_analysis['search_type'] == 'person'):
            name_search = query_analysis.get('direct_name', '')
            if not name_search and 'entities' in query_analysis:
                # Извлекаем имя из сущностей, если оно не было определено напрямую
                for entity in query_analysis['entities']:
                    if isinstance(entity, str) and entity[0].isupper():
                        name_search = entity
                        break
            
            if name_search:
                logger.info(f"Прямой поиск по имени: {name_search}")
                
                # Ищем профиль с таким именем
                name_results = []
                for profile in self.profiles:
                    if name_search.lower() in profile['name'].lower():
                        name_results.append(profile)
                
                if name_results:
                    logger.info(f"Найден профиль по имени: {name_results[0]['name']}")
                    return name_results[:limit]
                    
        results = self._perform_vector_search(query_analysis, query_for_vector, limit)
        end_time = time.time()
        logger.info(f"Поиск выполнен за {end_time - start_time:.3f}с. Найдено {len(results)} результатов")
        return results
    
    def _perform_vector_search(self, query_analysis, query_for_vector, limit):
        """Выполняет векторный поиск по подготовленному запросу"""
        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query_for_vector, normalize_embeddings=True)
        
        # Вычисляем косинусное сходство со всеми профилями
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Также выполняем простой текстовый поиск
        text_scores = np.zeros(len(self.profiles))
        query_terms = query_for_vector.lower().split()
        
        for i, profile in enumerate(self.profiles):
            content = profile['content'].lower()
            score = sum(1 for term in query_terms if term in content)
            text_scores[i] = score / len(query_terms) if query_terms else 0
            
        # Комбинируем векторный и текстовый поиск
        combined_scores = similarities * self.vector_weight + text_scores * (1 - self.vector_weight)
        
        # Сортируем по убыванию релевантности
        top_indices = np.argsort(combined_scores)[::-1][:limit*2]  # Берем двойной лимит для Mistral-ранжирования
        
        # Формируем результаты
        results = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            if score > 0:  # Только положительные оценки
                results.append({
                    **self.profiles[idx],
                    'similarity': score
                })
        
        # Если есть Mistral и есть результаты, переранжируем их
        if self.use_mistral and self._mistral_enhancer and results:
            try:
                ranked_results = self._mistral_enhancer.rerank_results(query_for_vector, results, top_n=limit)
                if ranked_results:
                    results = ranked_results
            except Exception as e:
                logger.error(f"Ошибка при переранжировании через Mistral: {e}")
                # В случае ошибки используем обычные результаты, ограниченные лимитом
                results = results[:limit]
        else:
            # Ограничиваем результаты без Mistral
            results = results[:limit]
                
        return results
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Анализирует поисковый запрос для определения его категории и особенностей"""
        analysis = {
            'original_query': query,
            'categories': [],
            'direct_name': None
        }
        
        # Проверка на прямой поиск по имени
        for pattern in self.query_patterns['name_search']:
            match = pattern.search(query)
            if match:
                name = match.group(1).strip()
                if len(name.split()) >= 1:  # Имя и фамилия или только имя
                    analysis['direct_name'] = name
                    analysis['categories'].append('name_search')
                    break
        
        # Проверка на поиск по навыкам
        for pattern in self.query_patterns['skills_search']:
            match = pattern.search(query)
            if match:
                skills = match.group(1).strip()
                analysis['skills'] = skills
                analysis['categories'].append('skills_search')
                break
        
        # Проверка на поиск по опыту
        for pattern in self.query_patterns['experience_search']:
            match = pattern.search(query)
            if match:
                experience = match.group(1).strip()
                analysis['experience'] = experience
                analysis['categories'].append('experience_search')
                break
        
        # Проверка на поиск по образованию
        for pattern in self.query_patterns['education_search']:
            match = pattern.search(query)
            if match:
                education = match.group(1).strip()
                analysis['education'] = education
                analysis['categories'].append('education_search')
                break
        
        # Если категории не определены, считаем запрос общим
        if not analysis['categories']:
            analysis['categories'].append('general_search')
        
        return analysis 