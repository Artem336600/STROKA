import os
import re
import json
import yaml
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from db_manager import ProfileDBManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("profile-loader")

class ProfileLoader:
    """
    Класс для загрузки профилей из файлов и импорта их в базу данных
    с преобразованием в векторные эмбеддинги
    """
    
    def __init__(
        self, 
        db_manager: ProfileDBManager,
        embedding_model_name: str = "distiluse-base-multilingual-cased-v2",
        batch_size: int = 8
    ):
        self.db_manager = db_manager
        self.model_name = embedding_model_name
        self.batch_size = batch_size
        self._embedder = None
        
        # Поддерживаемые форматы файлов и их обработчики
        self.supported_formats = {
            ".txt": self._process_text_file,
            ".json": self._process_json_file,
            ".md": self._process_markdown_file,
            ".yaml": self._process_yaml_file,
            ".yml": self._process_yaml_file
        }
        
        # Шаблоны для извлечения информации из профилей
        self.patterns = {
            "name": [
                re.compile(r'^(?:Имя|Name):\s*(.+)$', re.IGNORECASE),
                re.compile(r'^#\s*(.+)$')  # Markdown заголовок 1 уровня
            ],
            "age": [
                re.compile(r'^(?:Возраст|Age):\s*(\d+)$', re.IGNORECASE),
                re.compile(r'\*\*(?:Возраст|Age):\*\*\s*(\d+)', re.IGNORECASE)
            ]
        }
    
    @property
    def embedder(self) -> SentenceTransformer:
        """Ленивая инициализация модели эмбеддингов"""
        if self._embedder is None:
            logger.info(f"Инициализация модели эмбеддингов: {self.model_name}")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def import_directory(
        self, 
        directory_path: str, 
        recursive: bool = False,
        batch_import: bool = True
    ) -> Tuple[int, int]:
        """
        Импортирует все поддерживаемые файлы профилей из указанной директории
        
        Args:
            directory_path: Путь к директории с файлами профилей
            recursive: Если True, рекурсивно обрабатывает поддиректории
            batch_import: Если True, импортирует профили пакетно для ускорения
            
        Returns:
            Кортеж (общее количество файлов, успешно импортировано)
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Директория {directory_path} не существует или не является директорией")
            return (0, 0)
        
        logger.info(f"Начало импорта профилей из директории: {directory_path}")
        start_time = time.time()
        
        # Сбор всех файлов
        files_to_process = []
        if recursive:
            for ext in self.supported_formats:
                files_to_process.extend(list(directory.glob(f"**/*{ext}")))
        else:
            for ext in self.supported_formats:
                files_to_process.extend(list(directory.glob(f"*{ext}")))
        
        if not files_to_process:
            logger.warning(f"В директории {directory_path} не найдено поддерживаемых файлов")
            return (0, 0)
        
        logger.info(f"Найдено {len(files_to_process)} файлов для обработки")
        
        # Обработка файлов
        if batch_import:
            # Пакетный импорт
            profiles_batch = []
            batch_count = 0
            success_count = 0
            
            for i, file_path in enumerate(files_to_process):
                try:
                    ext = file_path.suffix.lower()
                    processor = self.supported_formats.get(ext)
                    
                    if processor:
                        profile_data = processor(file_path)
                        if profile_data:
                            # Добавляем путь к исходному файлу
                            profile_data['source_file'] = str(file_path)
                            profiles_batch.append(profile_data)
                            
                            # Импортируем пакет, когда достигаем batch_size
                            if len(profiles_batch) >= self.batch_size:
                                batch_count += 1
                                # Создаем эмбеддинги для пакета
                                self._add_embeddings_to_batch(profiles_batch)
                                # Импортируем пакет в БД
                                added = self.db_manager.batch_add_profiles(profiles_batch)
                                success_count += added
                                logger.info(f"Импортирован пакет {batch_count}, добавлено {added} профилей")
                                profiles_batch = []
                    
                    # Отображаем прогресс
                    if (i+1) % 10 == 0 or i+1 == len(files_to_process):
                        logger.info(f"Обработано {i+1}/{len(files_to_process)} файлов")
                
                except Exception as e:
                    logger.error(f"Ошибка обработки файла {file_path}: {e}")
            
            # Импортируем оставшиеся профили
            if profiles_batch:
                batch_count += 1
                self._add_embeddings_to_batch(profiles_batch)
                added = self.db_manager.batch_add_profiles(profiles_batch)
                success_count += added
                logger.info(f"Импортирован финальный пакет {batch_count}, добавлено {added} профилей")
        
        else:
            # Последовательный импорт
            success_count = 0
            for i, file_path in enumerate(files_to_process):
                try:
                    ext = file_path.suffix.lower()
                    processor = self.supported_formats.get(ext)
                    
                    if processor:
                        profile_data = processor(file_path)
                        if profile_data:
                            # Создаем эмбеддинг для профиля
                            embedding = self.create_embedding(profile_data['content'])
                            
                            # Импортируем профиль в БД
                            profile_id = self.db_manager.add_profile(
                                name=profile_data['name'],
                                content=profile_data['content'],
                                embedding=embedding,
                                metadata=profile_data.get('metadata', {}),
                                source_file=str(file_path)
                            )
                            
                            if profile_id > 0:
                                success_count += 1
                                logger.info(f"Импортирован профиль: {profile_data['name']} (id={profile_id})")
                    
                    # Отображаем прогресс
                    if (i+1) % 5 == 0 or i+1 == len(files_to_process):
                        logger.info(f"Обработано {i+1}/{len(files_to_process)} файлов")
                
                except Exception as e:
                    logger.error(f"Ошибка обработки файла {file_path}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Импорт завершен за {total_time:.2f} секунд. "
                   f"Всего: {len(files_to_process)}, успешно: {success_count}")
        
        return (len(files_to_process), success_count)
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Создает векторное представление текста профиля"""
        return self.embedder.encode(text, normalize_embeddings=True)
    
    def _add_embeddings_to_batch(self, profiles_batch: List[Dict[str, Any]]) -> None:
        """Добавляет эмбеддинги к пакету профилей"""
        if not profiles_batch:
            return
        
        # Извлекаем тексты для эмбеддингов
        texts = [profile['content'] for profile in profiles_batch]
        
        # Создаем эмбеддинги пакетом
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 10)
        
        # Добавляем эмбеддинги к соответствующим профилям
        for i, profile in enumerate(profiles_batch):
            profile['embedding'] = embeddings[i]
    
    def _process_text_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Обрабатывает текстовый файл профиля"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            logger.warning(f"Пустой файл: {file_path}")
            return None
        
        # Извлекаем имя из содержимого
        name = self._extract_name(content, file_path.stem)
        
        # Извлекаем метаданные
        metadata = self._extract_metadata(content)
        
        return {
            'name': name,
            'content': content,
            'metadata': metadata
        }
    
    def _process_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Обрабатывает JSON файл профиля"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Преобразуем JSON в текстовый формат
            if isinstance(data, dict):
                content = "\n".join([f"{key}: {value}" for key, value in data.items()])
                name = data.get("Имя", data.get("Name", file_path.stem))
                
                # Извлекаем метаданные
                metadata = {
                    'age': data.get("Возраст", data.get("Age")),
                    'skills': data.get("Навыки", data.get("Skills", "")),
                    'original_format': 'json'
                }
                
                return {
                    'name': name,
                    'content': content,
                    'metadata': metadata
                }
            else:
                logger.warning(f"Некорректный формат JSON в файле {file_path}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON в файле {file_path}: {e}")
            return None
    
    def _process_markdown_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Обрабатывает Markdown файл профиля"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            logger.warning(f"Пустой файл: {file_path}")
            return None
        
        # Извлекаем имя из заголовка Markdown
        name = self._extract_name(content, file_path.stem)
        
        # Преобразуем Markdown в текстовый формат
        text_content = re.sub(r'#+ ', '', content)  # Удаляем символы заголовка
        text_content = re.sub(r'\*\*(.*?)\*\*', r'\1', text_content)  # Удаляем выделение жирным
        
        # Извлекаем метаданные
        metadata = self._extract_metadata(content)
        metadata['original_format'] = 'markdown'
        
        return {
            'name': name,
            'content': text_content,
            'metadata': metadata
        }
    
    def _process_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Обрабатывает YAML файл профиля"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            # Преобразуем YAML в текстовый формат
            if isinstance(data, dict):
                content = "\n".join([f"{key}: {value}" for key, value in data.items()])
                name = data.get("Имя", data.get("Name", file_path.stem))
                
                # Извлекаем метаданные
                metadata = {
                    'age': data.get("Возраст", data.get("Age")),
                    'skills': data.get("Навыки", data.get("Skills", "")),
                    'original_format': 'yaml'
                }
                
                return {
                    'name': name,
                    'content': content,
                    'metadata': metadata
                }
            else:
                logger.warning(f"Некорректный формат YAML в файле {file_path}")
                return None
                
        except yaml.YAMLError as e:
            logger.error(f"Ошибка декодирования YAML в файле {file_path}: {e}")
            return None
    
    def _extract_name(self, content: str, default_name: str) -> str:
        """Извлекает имя из содержимого файла"""
        # Проверяем различные шаблоны для имени
        for pattern in self.patterns['name']:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        
        # Если не удалось найти, используем имя файла
        return default_name
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Извлекает метаданные из содержимого файла"""
        metadata = {}
        
        # Извлекаем возраст
        for pattern in self.patterns['age']:
            match = pattern.search(content)
            if match:
                try:
                    metadata['age'] = int(match.group(1))
                    break
                except ValueError:
                    pass
        
        # Извлекаем навыки
        skills_match = re.search(r'(?:Навыки|Skills):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if skills_match:
            skills_text = skills_match.group(1).strip()
            # Разделяем навыки по запятым
            metadata['skills'] = [skill.strip() for skill in skills_text.split(',')]
        
        # Извлекаем образование
        education_match = re.search(r'(?:Образование|Education):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if education_match:
            metadata['education'] = education_match.group(1).strip()
        
        # Извлекаем опыт работы
        experience_match = re.search(r'(?:Опыт работы|Experience):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if experience_match:
            metadata['experience'] = experience_match.group(1).strip()
        
        return metadata


# Пример использования
if __name__ == "__main__":
    # Подключение к базе данных
    db_manager = ProfileDBManager(
        db_name="people_profiles",
        db_user="postgres",
        db_password="postgres"
    )
    
    if db_manager.connect() and db_manager.initialize_db():
        # Создание загрузчика профилей
        loader = ProfileLoader(db_manager)
        
        # Импорт профилей из директории
        total, success = loader.import_directory("people", recursive=False)
        
        print(f"Импортировано {success} из {total} профилей")
        print(f"Всего профилей в базе: {db_manager.count_profiles()}")
        
        # Закрытие соединения
        db_manager.close() 