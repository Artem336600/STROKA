"""
Утилиты для поисковой системы профилей
"""
import os
import re
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def show_welcome_message():
    """Выводит приветственное сообщение и инструкции"""
    welcome = """
    ╔════════════════════════════════════════════════════════╗
    ║                  ПОИСКОВАЯ СИСТЕМА                     ║
    ║                 ПРОФИЛЕЙ СПЕЦИАЛИСТОВ                  ║
    ║                 (STROKA)                               ║
    ╚════════════════════════════════════════════════════════╝
    
    Инструкции:
    - Введите поисковый запрос, чтобы найти подходящих специалистов
    - Примеры запросов:
      * Найти программистов со знанием Python
      * Кто имеет опыт работы в банках более 5 лет
      * Инженеры с образованием МГТУ
      * Информация о Иване Петрове
    
    Специальные команды:
    - :quit, :q или :выход - завершить программу
    - :help или :помощь - показать эту инструкцию
    """
    print(welcome)

def extract_metadata(content: str) -> Dict[str, Any]:
    """Извлекает метаданные из содержимого файла"""
    metadata = {}
    
    # Извлекаем возраст
    age_match = re.search(r'(?:Возраст|Age):\s*(\d+)', content, re.IGNORECASE)
    if age_match:
        try:
            metadata['age'] = int(age_match.group(1))
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

def extract_name(content: str, default_name: str) -> str:
    """Извлекает имя из содержимого файла"""
    name_match = re.search(r'^(?:Имя|Name):\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
    if name_match:
        return name_match.group(1).strip()
    
    # Проверяем Markdown заголовок
    name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if name_match:
        return name_match.group(1).strip()
        
    # Если не удалось найти, используем имя файла
    return default_name

def load_config(config_path="config.json"):
    """Загружает конфигурацию из файла или использует значения по умолчанию"""
    default_config = {
        "embedding_model": "distiluse-base-multilingual-cased-v2",
        "data_dir": "people/",
        "vector_weight": 0.7,
        "search_limit": 7
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # Обновляем настройки из пользовательского конфига
            for key, value in user_config.items():
                default_config[key] = value
                
            logging.info(f"Конфигурация загружена из {config_path}")
        except Exception as e:
            logging.error(f"Ошибка при загрузке конфигурации: {e}")
            logging.info("Используются настройки по умолчанию")
    else:
        logging.info(f"Файл конфигурации {config_path} не найден, используются настройки по умолчанию")
    
    return default_config 