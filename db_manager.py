import os
import json
import logging
import numpy as np
import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db-manager")

class ProfileDBManager:
    """Менеджер для работы с базой данных профилей с поддержкой векторного поиска"""
    
    def __init__(
        self, 
        db_name: str = "people_profiles",
        db_user: str = "postgres",
        db_password: str = "postgres",
        db_host: str = "localhost",
        db_port: str = "5432",
        vector_dim: int = 384  # Размерность эмбеддингов (зависит от используемой модели)
    ):
        """Инициализация подключения к БД PostgreSQL с pgvector"""
        self.connection_params = {
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port
        }
        self.vector_dim = vector_dim
        self.conn = None
        self.is_initialized = False
        
    def connect(self) -> bool:
        """Установка соединения с базой данных"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = False
            logger.info(f"Успешное подключение к БД {self.connection_params['dbname']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            return False
    
    def initialize_db(self) -> bool:
        """Инициализация базы данных и таблиц"""
        if not self.conn:
            if not self.connect():
                return False
                
        try:
            with self.conn.cursor() as cursor:
                # Установка расширения pgvector
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Создание таблицы профилей
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(%s),
                    source_file VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """ % self.vector_dim)
                
                # Создание индекса для полнотекстового поиска
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS profiles_content_idx ON profiles 
                USING gin(to_tsvector('russian', content));
                """)
                
                # Создание индекса для векторного поиска (HNSW работает быстрее для больших коллекций)
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS profiles_embedding_idx ON profiles 
                USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Создание таблицы для логов поиска
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_logs (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    results_count INT,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                self.conn.commit()
                self.is_initialized = True
                logger.info("База данных успешно инициализирована")
                return True
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Ошибка инициализации БД: {e}")
            return False
    
    def add_profile(
        self, 
        name: str, 
        content: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any] = None,
        source_file: str = None
    ) -> int:
        """Добавление профиля в базу данных"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                # Проверяем существование профиля с таким именем
                cursor.execute("SELECT id FROM profiles WHERE name = %s", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Обновляем существующий профиль
                    cursor.execute("""
                    UPDATE profiles 
                    SET content = %s, embedding = %s, metadata = %s, 
                        source_file = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s
                    RETURNING id;
                    """, (content, embedding.tolist(), json.dumps(metadata or {}), source_file, name))
                    profile_id = cursor.fetchone()[0]
                    self.conn.commit()
                    logger.info(f"Обновлен профиль: {name} (id={profile_id})")
                else:
                    # Добавляем новый профиль
                    cursor.execute("""
                    INSERT INTO profiles (name, content, embedding, metadata, source_file)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """, (name, content, embedding.tolist(), json.dumps(metadata or {}), source_file))
                    profile_id = cursor.fetchone()[0]
                    self.conn.commit()
                    logger.info(f"Добавлен новый профиль: {name} (id={profile_id})")
                
                return profile_id
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Ошибка добавления профиля {name}: {e}")
            return -1
    
    def batch_add_profiles(
        self, 
        profiles: List[Dict[str, Any]]
    ) -> int:
        """Пакетное добавление профилей в базу данных"""
        if not self.is_initialized:
            self.initialize_db()
            
        added_count = 0
        try:
            with self.conn.cursor() as cursor:
                # Подготавливаем данные для пакетной вставки
                args = []
                for profile in profiles:
                    name = profile.get('name', '')
                    content = profile.get('content', '')
                    embedding = profile.get('embedding')
                    metadata = profile.get('metadata', {})
                    source_file = profile.get('source_file', '')
                    
                    if not name or not content or embedding is None:
                        logger.warning(f"Пропущен некорректный профиль: {name}")
                        continue
                        
                    args.append((
                        name, content, embedding.tolist(), 
                        json.dumps(metadata), source_file
                    ))
                
                # Используем технику upsert для обновления/вставки
                psycopg2.extras.execute_batch(cursor, """
                INSERT INTO profiles (name, content, embedding, metadata, source_file)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    source_file = EXCLUDED.source_file,
                    updated_at = CURRENT_TIMESTAMP
                """, args)
                
                added_count = len(args)
                self.conn.commit()
                logger.info(f"Пакетное добавление: обработано {added_count} профилей")
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Ошибка пакетного добавления профилей: {e}")
            
        return added_count
    
    def search_by_vector(
        self, 
        embedding: np.ndarray, 
        limit: int = 5, 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Поиск профилей по векторному эмбеддингу"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                # Выполняем векторный поиск с косинусным расстоянием
                cursor.execute("""
                SELECT 
                    id, name, content, metadata, source_file,
                    1 - (embedding <=> %s) as similarity
                FROM profiles 
                WHERE 1 - (embedding <=> %s) > %s
                ORDER BY similarity DESC
                LIMIT %s;
                """, (embedding.tolist(), embedding.tolist(), threshold, limit))
                
                results = []
                for row in cursor.fetchall():
                    profile_id, name, content, metadata_json, source_file, similarity = row
                    results.append({
                        'id': profile_id,
                        'name': name,
                        'content': content,
                        'metadata': json.loads(metadata_json) if metadata_json else {},
                        'source_file': source_file,
                        'similarity': similarity
                    })
                
                return results
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка векторного поиска: {e}")
            return []
    
    def search_by_text(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Полнотекстовый поиск профилей"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                # Подготавливаем запрос для полнотекстового поиска
                # Используем to_tsquery с учетом русского языка
                search_query = ' & '.join(query.split())
                
                cursor.execute("""
                SELECT 
                    id, name, content, metadata, source_file,
                    ts_rank_cd(to_tsvector('russian', content), to_tsquery('russian', %s)) as rank
                FROM profiles 
                WHERE to_tsvector('russian', content) @@ to_tsquery('russian', %s)
                ORDER BY rank DESC
                LIMIT %s;
                """, (search_query, search_query, limit))
                
                results = []
                for row in cursor.fetchall():
                    profile_id, name, content, metadata_json, source_file, rank = row
                    results.append({
                        'id': profile_id,
                        'name': name,
                        'content': content,
                        'metadata': json.loads(metadata_json) if metadata_json else {},
                        'source_file': source_file,
                        'text_rank': rank
                    })
                
                return results
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка текстового поиска: {e}")
            return []
    
    def hybrid_search(
        self, 
        text_query: str,
        embedding: np.ndarray,
        limit: int = 5,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Гибридный поиск, объединяющий текстовый и векторный"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                # Подготавливаем запрос для полнотекстового поиска
                search_tokens = ' & '.join(text_query.split())
                
                # Гибридный запрос с взвешиванием текстового и векторного поиска
                cursor.execute("""
                WITH vector_search AS (
                    SELECT 
                        id, 
                        1 - (embedding <=> %s) as v_score
                    FROM profiles
                ),
                text_search AS (
                    SELECT 
                        id, 
                        ts_rank_cd(to_tsvector('russian', content), to_tsquery('russian', %s)) as t_score
                    FROM profiles 
                    WHERE to_tsvector('russian', content || ' ' || name) @@ to_tsquery('russian', %s)
                )
                SELECT 
                    p.id, p.name, p.content, p.metadata, p.source_file,
                    COALESCE(v.v_score, 0) * %s + COALESCE(t.t_score, 0) * %s as hybrid_score
                FROM profiles p
                LEFT JOIN vector_search v ON p.id = v.id
                LEFT JOIN text_search t ON p.id = t.id
                WHERE v.v_score IS NOT NULL OR t.t_score IS NOT NULL
                ORDER BY hybrid_score DESC
                LIMIT %s;
                """, (
                    embedding.tolist(), 
                    search_tokens, search_tokens,
                    vector_weight, 1 - vector_weight,
                    limit
                ))
                
                results = []
                for row in cursor.fetchall():
                    profile_id, name, content, metadata_json, source_file, score = row
                    results.append({
                        'id': profile_id,
                        'name': name,
                        'content': content,
                        'metadata': json.loads(metadata_json) if metadata_json else {},
                        'source_file': source_file,
                        'score': score
                    })
                
                # Сохраняем запрос в логи
                self._log_search(text_query, len(results))
                
                return results
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка гибридного поиска: {e}")
            return []
    
    def get_profile_by_id(self, profile_id: int) -> Optional[Dict[str, Any]]:
        """Получение профиля по ID"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT id, name, content, metadata, source_file
                FROM profiles 
                WHERE id = %s
                """, (profile_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                profile_id, name, content, metadata_json, source_file = row
                return {
                    'id': profile_id,
                    'name': name,
                    'content': content,
                    'metadata': json.loads(metadata_json) if metadata_json else {},
                    'source_file': source_file
                }
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка получения профиля {profile_id}: {e}")
            return None
    
    def get_profile_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Получение профиля по имени"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                SELECT id, name, content, metadata, source_file
                FROM profiles 
                WHERE name ILIKE %s
                """, (f"%{name}%",))
                
                row = cursor.fetchone()
                if not row:
                    return None
                    
                profile_id, name, content, metadata_json, source_file = row
                return {
                    'id': profile_id,
                    'name': name,
                    'content': content,
                    'metadata': json.loads(metadata_json) if metadata_json else {},
                    'source_file': source_file
                }
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка получения профиля по имени {name}: {e}")
            return None
    
    def delete_profile(self, profile_id: int) -> bool:
        """Удаление профиля из базы данных"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM profiles WHERE id = %s", (profile_id,))
                affected = cursor.rowcount
                self.conn.commit()
                
                if affected > 0:
                    logger.info(f"Удален профиль с ID {profile_id}")
                    return True
                else:
                    logger.warning(f"Профиль с ID {profile_id} не найден")
                    return False
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Ошибка удаления профиля {profile_id}: {e}")
            return False
    
    def count_profiles(self) -> int:
        """Получение количества профилей в базе данных"""
        if not self.is_initialized:
            self.initialize_db()
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM profiles")
                return cursor.fetchone()[0]
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка подсчета профилей: {e}")
            return 0
    
    def _log_search(self, query: str, results_count: int, processing_time: float = None) -> None:
        """Сохранение информации о поисковом запросе"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO search_logs (query, results_count, processing_time)
                VALUES (%s, %s, %s)
                """, (query, results_count, processing_time))
                self.conn.commit()
                
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Ошибка логирования поиска: {e}")
    
    def get_search_stats(self, limit: int = 10) -> Dict[str, Any]:
        """Получение статистики поисковых запросов"""
        if not self.is_initialized:
            self.initialize_db()
            
        stats = {
            'total_searches': 0,
            'avg_results': 0,
            'avg_time': 0,
            'recent_queries': [],
            'popular_queries': []
        }
        
        try:
            with self.conn.cursor() as cursor:
                # Общее количество запросов
                cursor.execute("SELECT COUNT(*) FROM search_logs")
                stats['total_searches'] = cursor.fetchone()[0]
                
                # Средние показатели
                cursor.execute("""
                SELECT 
                    AVG(results_count) as avg_results,
                    AVG(processing_time) as avg_time
                FROM search_logs
                """)
                avg_results, avg_time = cursor.fetchone()
                stats['avg_results'] = round(avg_results or 0, 2)
                stats['avg_time'] = round(avg_time or 0, 3)
                
                # Последние запросы
                cursor.execute("""
                SELECT query, results_count, created_at
                FROM search_logs
                ORDER BY created_at DESC
                LIMIT %s
                """, (limit,))
                
                stats['recent_queries'] = [
                    {
                        'query': query,
                        'results': results_count,
                        'date': created_at.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for query, results_count, created_at in cursor.fetchall()
                ]
                
                # Популярные запросы
                cursor.execute("""
                SELECT query, COUNT(*) as count
                FROM search_logs
                GROUP BY query
                ORDER BY count DESC
                LIMIT %s
                """, (limit,))
                
                stats['popular_queries'] = [
                    {'query': query, 'count': count}
                    for query, count in cursor.fetchall()
                ]
                
                return stats
                
        except psycopg2.Error as e:
            logger.error(f"Ошибка получения статистики поиска: {e}")
            return stats
    
    def close(self) -> None:
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            logger.info("Соединение с базой данных закрыто")

# Пример использования
if __name__ == "__main__":
    # Тестирование функциональности
    db = ProfileDBManager()
    if db.connect() and db.initialize_db():
        print(f"Количество профилей в базе: {db.count_profiles()}")
        
        # Пример добавления профиля
        test_embedding = np.random.rand(384).astype(np.float32)  # Размерность должна соответствовать vector_dim
        test_profile = {
            'name': 'Тестовый профиль',
            'content': 'Это тестовый профиль для проверки работы базы данных',
            'metadata': {'age': 30, 'skills': ['Python', 'SQL']}
        }
        
        profile_id = db.add_profile(
            name=test_profile['name'],
            content=test_profile['content'],
            embedding=test_embedding,
            metadata=test_profile['metadata']
        )
        
        print(f"Добавлен профиль с ID: {profile_id}")
        db.close() 