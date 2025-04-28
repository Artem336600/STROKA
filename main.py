import os
import time
import json
import logging
import argparse
from pathlib import Path

from db_manager import ProfileDBManager
from profile_loader import ProfileLoader
from search_engine import ProfileSearchEngine

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("search-system")

# Настройки по умолчанию
DEFAULT_CONFIG = {
    "db": {
        "name": "people_profiles",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": 5432,
        "vector_dim": 512
    },
    "embedding_model": "distiluse-base-multilingual-cased-v2",
    "data_dir": "people/",
    "vector_weight": 0.7,
    "search_limit": 7
}

def load_config(config_path="config.json"):
    """Загружает конфигурацию из файла или использует значения по умолчанию"""
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # Обновляем настройки БД
            if 'db' in user_config:
                for key, value in user_config['db'].items():
                    config['db'][key] = value
                    
            # Обновляем другие настройки
            for key in ['embedding_model', 'data_dir', 'vector_weight', 'search_limit']:
                if key in user_config:
                    config[key] = user_config[key]
                    
            logger.info(f"Конфигурация загружена из {config_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            logger.info("Используются настройки по умолчанию")
    else:
        logger.info(f"Файл конфигурации {config_path} не найден, используются настройки по умолчанию")
    
    return config

def init_db_and_loader(config):
    """Инициализация базы данных и загрузчика профилей"""
    # Создаем менеджер БД
    db_manager = ProfileDBManager(
        db_name=config['db']['name'],
        db_user=config['db']['user'],
        db_password=config['db']['password'],
        db_host=config['db']['host'],
        db_port=config['db']['port'],
        vector_dim=config['db']['vector_dim']
    )
    
    # Подключаемся к БД и инициализируем структуру
    if not db_manager.connect():
        logger.error("Не удалось подключиться к базе данных")
        return None, None
    
    if not db_manager.initialize_db():
        logger.error("Не удалось инициализировать структуру базы данных")
        return None, None
    
    # Создаем загрузчик профилей
    profile_loader = ProfileLoader(
        db_manager=db_manager,
        embedding_model_name=config['embedding_model']
    )
    
    return db_manager, profile_loader

def import_profiles(loader, data_dir, force_reimport=False):
    """Импортирует профили из директории в базу данных"""
    if not os.path.exists(data_dir):
        logger.error(f"Директория с данными не найдена: {data_dir}")
        return False
    
    try:
        # Проверяем наличие профилей в БД
        profile_count = loader.db_manager.count_profiles()
        if profile_count > 0 and not force_reimport:
            logger.info(f"В базе данных уже есть {profile_count} профилей. Используйте --reimport для повторного импорта")
            return True
        
        # Импортируем профили
        start_time = time.time()
        imported_count = loader.import_directory(data_dir, recursive=True)
        end_time = time.time()
        
        logger.info(f"Импортировано {imported_count} профилей за {end_time - start_time:.2f}с")
        return True
    except Exception as e:
        logger.error(f"Ошибка при импорте профилей: {e}")
        return False

def show_welcome_message():
    """Выводит приветственное сообщение и инструкции"""
    welcome = """
    ╔════════════════════════════════════════════════════════╗
    ║                  ПОИСКОВАЯ СИСТЕМА                     ║
    ║                 ПРОФИЛЕЙ СПЕЦИАЛИСТОВ                  ║
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
    - :stats или :статистика - показать статистику поисковых запросов
    - :import или :импорт - повторно импортировать профили
    """
    print(welcome)

def show_search_stats(search_engine):
    """Выводит статистику поисковых запросов"""
    stats = search_engine.get_search_stats()
    
    print("\n╔═ Статистика поисковых запросов ═════════════════╗")
    print(f"║ Всего запросов: {stats.get('total_queries', 0):<31} ║")
    print("╠═════════════════════════════════════════════════╣")
    
    if 'popular_queries' in stats and stats['popular_queries']:
        print("║ Популярные запросы:                            ║")
        for i, (query, count) in enumerate(stats['popular_queries'][:5], 1):
            print(f"║ {i}. \"{query}\" ({count} раз) {' ' * (28 - len(query) - len(str(count)))}║")
    else:
        print("║ Нет данных о популярных запросах                ║")
    
    print("╚═════════════════════════════════════════════════╝\n")

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
            elif query.lower() in [':stats', ':статистика']:
                show_search_stats(search_engine)
                continue
            elif query.lower() in [':import', ':импорт']:
                print("Эта функция требует прав администратора")
                continue
            
            # Выполняем поиск
            print(f"Поиск: \"{query}\"")
            start_time = time.time()
            results = search_engine.search(query)
            end_time = time.time()
            
            # Выводим результаты
            if results:
                print(f"\nНайдено результатов: {len(results)} [{end_time - start_time:.3f}с]")
                
                for i, result in enumerate(results[:5], 1):
                    name = result.get('name', 'Без имени')
                    similarity = result.get('similarity', result.get('score', 0)) * 100
                    print(f"{i}. {name} [релевантность: {similarity:.1f}%]")
                
                if len(results) > 5:
                    print(f"... и ещё {len(results) - 5} результатов")
                
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
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Поисковая система профилей специалистов')
    parser.add_argument('--config', default='config.json', help='Путь к файлу конфигурации')
    parser.add_argument('--reimport', action='store_true', help='Принудительно реимпортировать профили')
    parser.add_argument('--query', help='Выполнить поиск и завершить работу')
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Инициализация компонентов
    db_manager, profile_loader = init_db_and_loader(config)
    if not db_manager or not profile_loader:
        logger.critical("Не удалось инициализировать компоненты системы")
        return
    
    # Импорт профилей
    if not import_profiles(profile_loader, config['data_dir'], args.reimport):
        logger.warning("Возникли проблемы при импорте профилей")
    
    # Создание поисковой системы
    search_engine = ProfileSearchEngine(
        db_manager=db_manager,
        embedding_model_name=config['embedding_model'],
        vector_search_weight=config['vector_weight'],
        default_limit=config['search_limit']
    )
    
    try:
        # Режим одиночного запроса или интерактивный режим
        if args.query:
            results = search_engine.search(args.query)
            if results:
                print(f"Найдено результатов: {len(results)}")
                for i, result in enumerate(results[:5], 1):
                    name = result.get('name', 'Без имени')
                    similarity = result.get('similarity', result.get('score', 0)) * 100
                    print(f"{i}. {name} [релевантность: {similarity:.1f}%]")
            else:
                print("Подходящие профили не найдены")
        else:
            interactive_search_loop(search_engine)
    finally:
        # Закрытие соединения с БД
        db_manager.close()
        logger.info("Работа программы завершена")

if __name__ == "__main__":
    main()
