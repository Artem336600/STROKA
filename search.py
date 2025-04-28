"""
Модуль для поиска по профилям
"""
import re
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("search")

class Searcher:
    """Класс для поиска по профилям"""
    
    def __init__(
        self, 
        profiles: List[Dict[str, Any]], 
        embedding_model_name: str = "distiluse-base-multilingual-cased-v2",
        vector_weight: float = 0.7,
        default_limit: int = 5
    ):
        self.profiles = profiles
        self.model_name = embedding_model_name
        self.vector_weight = vector_weight
        self.default_limit = default_limit
        self._embedder = None
        self.embeddings = None
        
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
        
        # Анализируем запрос
        query_analysis = self.analyze_query(query)
        
        # Прямой поиск по имени
        if query_analysis.get('direct_name'):
            name_search = query_analysis.get('direct_name')
            logger.info(f"Прямой поиск по имени: {name_search}")
            
            # Ищем профиль с таким именем
            name_results = []
            for profile in self.profiles:
                if name_search.lower() in profile['name'].lower():
                    name_results.append(profile)
            
            if name_results:
                logger.info(f"Найден профиль по имени: {name_results[0]['name']}")
                return name_results[:limit]
        
        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        
        # Вычисляем косинусное сходство со всеми профилями
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Также выполняем простой текстовый поиск
        text_scores = np.zeros(len(self.profiles))
        query_terms = query.lower().split()
        
        for i, profile in enumerate(self.profiles):
            content = profile['content'].lower()
            score = sum(1 for term in query_terms if term in content)
            text_scores[i] = score / len(query_terms) if query_terms else 0
            
        # Комбинируем векторный и текстовый поиск
        combined_scores = similarities * self.vector_weight + text_scores * (1 - self.vector_weight)
        
        # Сортируем по убыванию релевантности
        top_indices = np.argsort(combined_scores)[::-1][:limit]
        
        # Формируем результаты
        results = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            if score > 0:  # Только положительные оценки
                results.append({
                    **self.profiles[idx],
                    'similarity': score
                })
                
        end_time = time.time()
        logger.info(f"Поиск выполнен за {end_time - start_time:.3f}с. Найдено {len(results)} результатов")
        
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