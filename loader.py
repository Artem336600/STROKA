"""
Модуль для загрузки профилей из файлов разных форматов
"""
import os
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from utils import extract_name, extract_metadata

logger = logging.getLogger("loader")

def load_profiles(data_dir: str) -> List[Dict[str, Any]]:
    """Загружает все профили из указанной директории"""
    profiles = []
    directory = Path(data_dir)
    
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Директория {data_dir} не существует")
        return profiles
    
    # Поддерживаемые форматы и их обработчики
    processors = {
        ".txt": process_text_file,
        ".json": process_json_file,
        ".md": process_markdown_file,
        ".yaml": process_yaml_file,
        ".yml": process_yaml_file
    }
    
    # Собираем все файлы поддерживаемых форматов
    files_to_process = []
    for ext in processors:
        files_to_process.extend(list(directory.glob(f"*{ext}")))
    
    logger.info(f"Найдено {len(files_to_process)} файлов для обработки")
    loaded_count = 0
    
    # Обрабатываем каждый файл
    for file_path in files_to_process:
        try:
            ext = file_path.suffix.lower()
            processor = processors.get(ext)
            
            if processor:
                profile_data = processor(file_path)
                if profile_data:
                    profiles.append(profile_data)
                    loaded_count += 1
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
    
    logger.info(f"Загружено {loaded_count} профилей")
    return profiles

def process_text_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Обрабатывает текстовый файл"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        # Извлекаем имя и метаданные
        name = extract_name(content, file_path.stem)
        metadata = extract_metadata(content)
        
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path),
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке текстового файла {file_path}: {e}")
        return None

def process_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Обрабатывает JSON файл"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return None
        
        # Преобразуем JSON в текстовый формат
        content = "\n".join([f"{key}: {value}" for key, value in data.items()])
        name = data.get("Имя", data.get("Name", file_path.stem))
        
        # Создаем метаданные
        metadata = {
            'age': data.get("Возраст", data.get("Age")),
            'skills': data.get("Навыки", data.get("Skills", "")),
            'education': data.get("Образование", data.get("Education")),
            'experience': data.get("Опыт работы", data.get("Experience")),
            'original_format': 'json'
        }
        
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path),
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке JSON файла {file_path}: {e}")
        return None

def process_markdown_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Обрабатывает Markdown файл"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        # Извлекаем имя и метаданные
        name = extract_name(content, file_path.stem)
        metadata = extract_metadata(content)
        metadata['original_format'] = 'markdown'
        
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path),
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке Markdown файла {file_path}: {e}")
        return None

def process_yaml_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Обрабатывает YAML файл"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            return None
        
        # Преобразуем YAML в текстовый формат
        content = "\n".join([f"{key}: {value}" for key, value in data.items()])
        name = data.get("Имя", data.get("Name", file_path.stem))
        
        # Создаем метаданные
        metadata = {
            'age': data.get("Возраст", data.get("Age")),
            'skills': data.get("Навыки", data.get("Skills", "")),
            'education': data.get("Образование", data.get("Education")),
            'experience': data.get("Опыт работы", data.get("Experience")),
            'original_format': 'yaml'
        }
        
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path),
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке YAML файла {file_path}: {e}")
        return None 