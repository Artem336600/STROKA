import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
import re

from db_manager import ProfileDBManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("search-engine")

class ProfileSearchEngine:
    """Поисковая система для профилей людей с использованием векторного поиска в PostgreSQL"""
    
    def __init__(
        self,
        db_manager: ProfileDBManager,
        embedding_model_name: str = "distiluse-base-multilingual-cased-v2",
        vector_search_weight: float = 0.7,
        default_limit: int = 7
    ):
        """
        Инициализация поисковой системы
        
        Args:
            db_manager: Менеджер для работы с базой данных
            embedding_model_name: Название модели для создания эмбеддингов
            vector_search_weight: Вес векторного поиска в гибридном поиске (0-1)
            default_limit: Количество результатов по умолчанию
        """
        self.db_manager = db_manager
        self.model_name = embedding_model_name
        self.vector_weight = vector_search_weight
        self.default_limit = default_limit
        self._embedder = None
        
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
    
    @property
    def embedder(self) -> SentenceTransformer:
        """Ленивая инициализация модели эмбеддингов"""
        if self._embedder is None:
            logger.info(f"Инициализация модели эмбеддингов: {self.model_name}")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def search(
        self, 
        query: str, 
        limit: int = None, 
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Выполняет поиск профилей по запросу
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            use_hybrid: Использовать гибридный поиск (семантический + текстовый)
            
        Returns:
            Список профилей, соответствующих запросу
        """
        if not query.strip():
            logger.warning("Пустой поисковый запрос")
            return []
        
        limit = limit or self.default_limit
        start_time = time.time()
        
        # Предварительный анализ запроса
        query_analysis = self.analyze_query(query)
        
        # Если в запросе явно указано имя, сначала ищем по имени
        if query_analysis.get('direct_name'):
            direct_name_search = query_analysis.get('direct_name')
            logger.info(f"Прямой поиск по имени: {direct_name_search}")
            
            # Получаем профиль по имени
            profile = self.db_manager.get_profile_by_name(direct_name_search)
            if profile:
                # Находим похожие профили для контекста
                embedding = self.create_embedding(profile['content'])
                similar_profiles = self.db_manager.search_by_vector(
                    embedding=embedding,
                    limit=limit-1
                )
                
                # Добавляем найденный профиль в начало списка
                results = [profile] + [p for p in similar_profiles if p['id'] != profile['id']]
                
                end_time = time.time()
                logger.info(f"Поиск по имени выполнен за {end_time - start_time:.3f}с. Найдено {len(results)} результатов")
                return results[:limit]
        
        # Создание эмбеддинга запроса
        query_embedding = self.create_embedding(query)
        
        # Выбор типа поиска
        if use_hybrid:
            # Гибридный поиск (семантический + текстовый)
            results = self.db_manager.hybrid_search(
                text_query=query,
                embedding=query_embedding,
                limit=limit,
                vector_weight=self.vector_weight
            )
        else:
            # Только семантический поиск
            results = self.db_manager.search_by_vector(
                embedding=query_embedding,
                limit=limit
            )
        
        end_time = time.time()
        logger.info(f"Поиск выполнен за {end_time - start_time:.3f}с. Найдено {len(results)} результатов")
        return results
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Анализирует поисковый запрос для определения его категории и особенностей
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Словарь с анализом запроса
        """
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
                if len(name.split()) >= 2:  # Имя и фамилия
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
        
        logger.debug(f"Анализ запроса: {analysis}")
        return analysis
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Создает векторное представление текста"""
        return self.embedder.encode(text, normalize_embeddings=True)
    
    def format_search_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str = None,
        max_context_length: int = 5000
    ) -> str:
        """
        Форматирует результаты поиска для вывода пользователю или передачи в LLM
        
        Args:
            results: Список результатов поиска
            query: Исходный запрос (для контекста)
            max_context_length: Максимальная длина контекста
            
        Returns:
            Отформатированный текст результатов
        """
        if not results:
            return "Подходящие профили не найдены."
        
        formatted_results = []
        total_length = 0
        
        for i, result in enumerate(results):
            # Форматирование каждого профиля
            profile_text = f"## {result['name']}\n\n{result['content']}"
            
            # Добавляем метаинформацию
            if 'similarity' in result and result['similarity'] is not None:
                profile_text += f"\n\n[Релевантность: {result['similarity']:.2f}]"
            elif 'score' in result and result['score'] is not None:
                profile_text += f"\n\n[Релевантность: {result['score']:.2f}]"
            
            # Проверяем длину
            profile_length = len(profile_text.split())
            if total_length + profile_length > max_context_length:
                formatted_results.append("... (информация о других профилях опущена из-за ограничения длины) ...")
                break
            
            formatted_results.append(profile_text)
            total_length += profile_length
        
        # Объединяем результаты с разделителями
        return "\n\n" + "\n\n---\n\n".join(formatted_results)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Получает статистику поисковых запросов"""
        return self.db_manager.get_search_stats()


# Пример использования
if __name__ == "__main__":
    # Подключение к базе данных
    db_manager = ProfileDBManager(
        db_name="people_profiles",
        db_user="postgres",
        db_password="postgres"
    )
    
    if db_manager.connect() and db_manager.initialize_db():
        # Создание поисковой системы
        search_engine = ProfileSearchEngine(db_manager)
        
        # Примеры запросов
        test_queries = [
            "Кто знает Python и машинное обучение?",
            "Найти людей с опытом работы более 10 лет",
            "Инженеры с образованием МГТУ Баумана",
            "Информация о Сергее Волкове"
        ]
        
        for query in test_queries:
            print(f"\nЗапрос: {query}")
            results = search_engine.search(query)
            print(f"Найдено результатов: {len(results)}")
            
            if results:
                print(f"Первый результат: {results[0]['name']}")
        
        # Закрытие соединения
        db_manager.close() 