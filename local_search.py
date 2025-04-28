import os
import re
import json
import yaml
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local-search")

class LocalSearchEngine:
    """Поисковая система для профилей без использования БД"""
    
    def __init__(
        self, 
        data_dir: str = "people",
        embedding_model_name: str = "distiluse-base-multilingual-cased-v2",
        vector_weight: float = 0.7
    ):
        self.data_dir = data_dir
        self.model_name = embedding_model_name
        self.vector_weight = vector_weight
        self._embedder = None
        self.profiles = []
        self.embeddings = []
        
        # Поддерживаемые форматы файлов и их обработчики
        self.supported_formats = {
            ".txt": self._process_text_file,
            ".json": self._process_json_file,
            ".md": self._process_markdown_file,
            ".yaml": self._process_yaml_file,
            ".yml": self._process_yaml_file
        }
        
    @property
    def embedder(self) -> SentenceTransformer:
        """Ленивая инициализация модели эмбеддингов"""
        if self._embedder is None:
            logger.info(f"Инициализация модели эмбеддингов: {self.model_name}")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def load_profiles(self) -> int:
        """Загружает профили из директории"""
        directory = Path(self.data_dir)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Директория {self.data_dir} не существует")
            return 0
            
        logger.info(f"Загрузка профилей из директории {self.data_dir}...")
        
        # Собираем все файлы поддерживаемых форматов
        files_to_process = []
        for ext in self.supported_formats:
            files_to_process.extend(list(directory.glob(f"*{ext}")))
            
        if not files_to_process:
            logger.warning(f"В директории {self.data_dir} не найдено поддерживаемых файлов")
            return 0
            
        logger.info(f"Найдено {len(files_to_process)} файлов для обработки")
        
        # Обработка файлов
        loaded_count = 0
        for file_path in files_to_process:
            try:
                ext = file_path.suffix.lower()
                processor = self.supported_formats.get(ext)
                
                if processor:
                    profile_data = processor(file_path)
                    if profile_data and 'content' in profile_data:
                        self.profiles.append(profile_data)
                        loaded_count += 1
            except Exception as e:
                logger.error(f"Ошибка обработки файла {file_path}: {e}")
                
        # Создаем эмбеддинги для всех профилей
        if loaded_count > 0:
            self._create_embeddings()
            
        logger.info(f"Загружено {loaded_count} профилей")
        return loaded_count
    
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
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Выполняет поиск по профилям"""
        if not query.strip():
            logger.warning("Пустой поисковый запрос")
            return []
            
        if len(self.profiles) == 0 or len(self.embeddings) == 0:
            logger.warning("Профили не загружены")
            return []
            
        logger.info(f"Поиск: \"{query}\"")
        start_time = time.time()
        
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
    
    def _process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Обрабатывает текстовый файл"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            return None
            
        # Извлекаем имя из первой строки (обычно "Имя: ...")
        name = file_path.stem
        name_match = re.search(r'^(?:Имя|Name):\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path)
        }
    
    def _process_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Обрабатывает JSON файл"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                # Преобразуем JSON в текстовый формат
                content = "\n".join([f"{key}: {value}" for key, value in data.items()])
                name = data.get("Имя", data.get("Name", file_path.stem))
                
                return {
                    'name': name,
                    'content': content,
                    'source_file': str(file_path)
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Ошибка чтения JSON файла {file_path}: {e}")
            return None
    
    def _process_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Обрабатывает Markdown файл"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            return None
            
        # Извлекаем имя из заголовка
        name = file_path.stem
        name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
            
        return {
            'name': name,
            'content': content,
            'source_file': str(file_path)
        }
    
    def _process_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Обрабатывает YAML файл"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
            if isinstance(data, dict):
                # Преобразуем YAML в текстовый формат
                content = "\n".join([f"{key}: {value}" for key, value in data.items()])
                name = data.get("Имя", data.get("Name", file_path.stem))
                
                return {
                    'name': name,
                    'content': content,
                    'source_file': str(file_path)
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Ошибка чтения YAML файла {file_path}: {e}")
            return None

def show_welcome_message():
    """Выводит приветственное сообщение и инструкции"""
    welcome = """
    ╔════════════════════════════════════════════════════════╗
    ║                  ПОИСКОВАЯ СИСТЕМА                     ║
    ║                 ПРОФИЛЕЙ СПЕЦИАЛИСТОВ                  ║
    ║                 (ЛОКАЛЬНАЯ ВЕРСИЯ)                     ║
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

def interactive_search_loop(search_engine):
    """Основной цикл интерактивного поиска"""
    show_welcome_message()
    
    while True:
        try:
            query = input("\nВведите поисковый запрос: ").strip()
            
            if not query:
                continue
                
            # Обработка специальных команд
            if query.lower() in [':q', ':quit', ':выход']:
                print("Завершение работы...")
                break
            elif query.lower() in [':help', ':помощь']:
                show_welcome_message()
                continue
            
            # Выполняем поиск
            print(f"Поиск: \"{query}\"")
            start_time = time.time()
            results = search_engine.search(query)
            end_time = time.time()
            
            # Выводим результаты
            if results:
                print(f"\nНайдено результатов: {len(results)} [{end_time - start_time:.3f}с]")
                
                for i, result in enumerate(results, 1):
                    name = result.get('name', 'Без имени')
                    similarity = result.get('similarity', 0) * 100
                    print(f"{i}. {name} [релевантность: {similarity:.1f}%]")
                
                # Спрашиваем, какой профиль показать подробнее
                while True:
                    try:
                        choice = input("\nВведите номер профиля для подробной информации (или Enter для продолжения): ")
                        if not choice:
                            break
                            
                        choice = int(choice)
                        if 1 <= choice <= len(results):
                            profile = results[choice-1]
                            print(f"\n{'='*50}")
                            print(f"Имя: {profile.get('name', 'Не указано')}")
                            print(f"{'='*50}")
                            print(profile.get('content', 'Информация отсутствует'))
                            print(f"{'='*50}")
                        else:
                            print("Неверный номер профиля")
                    except ValueError:
                        print("Необходимо ввести число")
                    except Exception as e:
                        logger.error(f"Ошибка при отображении профиля: {e}")
            else:
                print("Подходящие профили не найдены")
                
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            print(f"Произошла ошибка: {e}")

def main():
    import argparse
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Локальная поисковая система профилей')
    parser.add_argument('--data-dir', default='people', help='Директория с профилями')
    parser.add_argument('--query', help='Выполнить поиск и завершить работу')
    args = parser.parse_args()
    
    # Создание поисковой системы
    search_engine = LocalSearchEngine(data_dir=args.data_dir)
    
    # Загрузка профилей
    loaded = search_engine.load_profiles()
    
    if loaded == 0:
        print("Профили не найдены. Проверьте директорию с данными.")
        return
    
    # Режим одиночного запроса или интерактивный режим
    if args.query:
        results = search_engine.search(args.query)
        if results:
            print(f"Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                name = result.get('name', 'Без имени')
                similarity = result.get('similarity', 0) * 100
                print(f"{i}. {name} [релевантность: {similarity:.1f}%]")
        else:
            print("Подходящие профили не найдены")
    else:
        interactive_search_loop(search_engine)

if __name__ == "__main__":
    main() 