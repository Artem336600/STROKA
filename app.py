"""
Основной модуль приложения для поиска профилей
"""
import argparse
import time
import logging
from typing import List, Dict, Any

from utils import show_welcome_message, load_config
from loader import load_profiles
from search import Searcher

logger = logging.getLogger("app")

def interactive_search_loop(searcher):
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
            results = searcher.search(query)
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
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Поисковая система профилей специалистов')
    parser.add_argument('--config', default='config.json', help='Путь к файлу конфигурации')
    parser.add_argument('--data-dir', help='Директория с профилями')
    parser.add_argument('--query', help='Выполнить поиск и завершить работу')
    parser.add_argument('--use-mistral', action='store_true', help='Использовать Mistral AI для улучшения поиска')
    parser.add_argument('--no-mistral', action='store_true', help='Не использовать Mistral AI')
    parser.add_argument('--mistral-model', default='mistral-small', help='Модель Mistral для использования')
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Если директория указана в аргументах, используем её
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    # Определяем, нужно ли использовать Mistral
    use_mistral = True  # По умолчанию включено
    if args.no_mistral:
        use_mistral = False
    elif args.use_mistral:
        use_mistral = True
    elif 'use_mistral' in config:
        use_mistral = config['use_mistral']
    
    # Модель Mistral
    mistral_model = args.mistral_model
    if 'mistral_model' in config:
        mistral_model = config.get('mistral_model', 'mistral-small')
    
    # Загрузка профилей
    profiles = load_profiles(config['data_dir'])
    
    if not profiles:
        print(f"Не найдены профили в директории {config['data_dir']}")
        return
    
    # Создаем поисковый движок
    searcher = Searcher(
        profiles=profiles,
        embedding_model_name=config['embedding_model'],
        vector_weight=config['vector_weight'],
        default_limit=config['search_limit'],
        use_mistral=use_mistral,
        mistral_model=mistral_model
    )
    
    if use_mistral:
        logger.info(f"Поиск с использованием Mistral AI (модель: {mistral_model})")
    else:
        logger.info("Поиск без использования Mistral AI")
    
    # Режим одиночного запроса или интерактивный режим
    if args.query:
        results = searcher.search(args.query)
        if results:
            print(f"Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                name = result.get('name', 'Без имени')
                similarity = result.get('similarity', 0) * 100
                print(f"{i}. {name} [релевантность: {similarity:.1f}%]")
        else:
            print("Подходящие профили не найдены")
    else:
        interactive_search_loop(searcher)

if __name__ == "__main__":
    main() 