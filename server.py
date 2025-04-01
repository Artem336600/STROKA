from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from supabase import create_client, Client
import jwt
from datetime import datetime, timedelta
import json
import random
import string

app = Flask(__name__, static_folder='.')
CORS(app)

# Инициализация Supabase
supabase_url = "https://ddfjcrfioaymllejalpm.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRkZmpjcmZpb2F5bWxsZWphbHBtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjQ3ODYzOSwiZXhwIjoyMDU4MDU0NjM5fQ.Dh42k1K07grKhF3DntbNLSwUifaXAa0Q6-LEIzRgpWM"
supabase: Client = create_client(supabase_url, supabase_key)

# Секретный ключ для JWT
JWT_SECRET = "your-secret-key-here"  # Замените на свой секретный ключ

# Инициализация клиента Mistral
api_key = "InDPitkUkV2JX5S1wdlWZwIfee6wTwLc"  # Ваш API ключ
client = MistralClient(api_key=api_key)

# База тегов, разбитая на категории
TAGS = {
    'программирование': [
        'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'go',
        'rust', 'scala', 'r', 'matlab', 'perl', 'haskell', 'lua', 'dart', 'typescript',
        'веб-разработка', 'фронтенд', 'бэкенд', 'fullstack', 'api', 'rest', 'graphql',
        'микросервисы', 'контейнеризация', 'docker', 'kubernetes', 'devops', 'ci/cd',
        'тестирование', 'unit-тесты', 'интеграционные тесты', 'e2e тесты', 'tdd', 'bdd',
        'базы данных', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
        'рефакторинг', 'паттерны проектирования', 'чистый код', 'архитектура'
    ],
    'дизайн': [
        'ui/ux', 'графический дизайн', 'веб-дизайн', 'интерфейсы', 'анимация',
        '3d-моделирование', 'иллюстрация', 'типография', 'логотипы', 'брендинг',
        'инфографика', 'моушн-дизайн', 'интерактивный дизайн', 'адаптивный дизайн',
        'материальный дизайн', 'плоский дизайн', 'минимализм', 'скетчинг', 'прототипирование',
        'фигма', 'adobe xd', 'sketch', 'photoshop', 'illustrator', 'after effects'
    ],
    'маркетинг': [
        'seo', 'контент-маркетинг', 'социальные сети', 'email-маркетинг', 'аналитика',
        'таргетинг', 'реклама', 'ppc', 'smm', 'контекстная реклама', 'медийная реклама',
        'вирусный маркетинг', 'inbound маркетинг', 'outbound маркетинг', 'affiliate маркетинг',
        'маркетинг влияния', 'контент-стратегия', 'копирайтинг', 'маркетинговая аналитика',
        'crm', 'маркетинговая автоматизация', 'a/b тестирование', 'конверсия', 'лидогенерация'
    ],
    'бизнес': [
        'стартапы', 'предпринимательство', 'инвестиции', 'финансы', 'управление проектами',
        'agile', 'scrum', 'kanban', 'лидерство', 'продажи', 'переговоры', 'клиентский сервис',
        'hr', 'рекрутинг', 'управление персоналом', 'стратегическое планирование',
        'бизнес-аналитика', 'управление рисками', 'бизнес-процессы', 'оптимизация',
        'консалтинг', 'коучинг', 'менторство', 'бизнес-планирование', 'финансовое планирование'
    ],
    'искусственный интеллект': [
        'машинное обучение', 'глубокое обучение', 'нейронные сети', 'компьютерное зрение',
        'обработка естественного языка', 'nlp', 'рекомендательные системы', 'большие данные',
        'data science', 'data mining', 'анализ данных', 'статистика', 'математика',
        'робототехника', 'автоматизация', 'искусственный интеллект', 'ai', 'ml', 'dl',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy'
    ],
    'безопасность': [
        'кибербезопасность', 'информационная безопасность', 'этичный хакинг',
        'penetration testing', 'security audit', 'криптография', 'ssl/tls',
        'firewall', 'антивирусы', 'vpn', '2fa', 'аутентификация', 'авторизация',
        'безопасность приложений', 'безопасность данных', 'privacy', 'gdpr',
        'социальная инженерия', 'фишинг', 'ransomware', 'ddos', 'xss', 'sql injection'
    ],
    'образование': [
        'онлайн-курсы', 'e-learning', 'дистанционное обучение', 'тренинги', 'вебинары',
        'мастер-классы', 'коучинг', 'менторство', 'самообразование', 'языки программирования',
        'иностранные языки', 'soft skills', 'hard skills', 'профессиональное развитие',
        'сертификации', 'дипломы', 'академическое образование', 'техническое образование',
        'гуманитарное образование', 'бизнес-образование', 'профессиональная переподготовка'
    ],
    'медиа': [
        'фотография', 'видеография', 'монтаж', 'цветокоррекция', 'звукозапись',
        'подкасты', 'стриминг', 'блогинг', 'vlog', 'контент-креация', 'социальные медиа',
        'youtube', 'instagram', 'tiktok', 'twitch', 'телевидение', 'радио', 'журналистика',
        'копирайтинг', 'редактирование', 'монтаж видео', 'анимация', 'моушн-графика'
    ],
    'финансы': [
        'инвестиции', 'трейдинг', 'криптовалюты', 'блокчейн', 'финансовые технологии',
        'банкинг', 'страхование', 'бухгалтерия', 'налогообложение', 'финансовое планирование',
        'управление активами', 'рынки капитала', 'forex', 'акции', 'облигации',
        'финансовая аналитика', 'риск-менеджмент', 'финансовое моделирование',
        'crowdfunding', 'p2p-кредитование', 'микрофинансирование'
    ],
    'здоровье': [
        'фитнес', 'йога', 'медитация', 'здоровое питание', 'диеты', 'спорт',
        'психология', 'ментальное здоровье', 'стресс-менеджмент', 'сон', 'релаксация',
        'физиотерапия', 'массаж', 'акупунктура', 'гомеопатия', 'натуропатия',
        'витамины', 'биодобавки', 'иммунитет', 'детокс', 'велнес', 'здоровый образ жизни'
    ],
    'творчество': [
        'рисование', 'живопись', 'скульптура', 'фотография', 'музыка', 'танцы',
        'писательство', 'поэзия', 'каллиграфия', 'handmade', 'декоративно-прикладное искусство',
        'дизайн интерьера', 'ландшафтный дизайн', 'флористика', 'hand lettering',
        'скрапбукинг', 'квиллинг', 'вышивка', 'вязание', 'шитье', 'декупаж'
    ],
    'путешествия': [
        'туризм', 'экотуризм', 'экстремальный туризм', 'гастрономический туризм',
        'культурный туризм', 'медицинский туризм', 'бэкпэкинг', 'кемпинг',
        'путешествия автостопом', 'круизы', 'сафари', 'горный туризм', 'дайвинг',
        'серфинг', 'альпинизм', 'пеший туризм', 'велотуризм', 'автомобильный туризм'
    ],
    'технологии': [
        'облачные технологии', 'iot', 'интернет вещей', '5g', '6g', 'квантовые вычисления',
        'edge computing', 'ар дополненная реальность', 'vr виртуальная реальность',
        '3d-печать', 'робототехника', 'беспилотные технологии', 'искусственный интеллект',
        'машинное обучение', 'большие данные', 'блокчейн', 'криптовалюты',
        'кибербезопасность', 'devops', 'микросервисы', 'контейнеризация'
    ]
}

# Создаем плоский список всех тегов
ALL_TAGS = [tag for category in TAGS.values() for tag in category]

# Хранилище кодов верификации
verification_codes = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/profile.html')
def profile():
    return send_from_directory('.', 'profile.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/styles.css')
def serve_styles():
    return send_from_directory('.', 'styles.css')

@app.route('/script.js')
def serve_script():
    return send_from_directory('.', 'script.js')

@app.route('/api/check-user', methods=['POST'])
def check_user():
    data = request.get_json()
    telegram = data.get('telegram')

    if not telegram:
        return jsonify({"error": "Telegram не указан"}), 400

    try:
        # Проверяем, существует ли пользователь
        existing_user = supabase.table('Stroka_test').select('*').eq('tg', telegram).execute()
        if existing_user.data:
            return jsonify({"error": "Пользователь с таким Telegram уже существует"}), 400
        
        return jsonify({"message": "Пользователь не существует"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    telegram = data.get('telegram')
    password = data.get('password')
    about = data.get('about')
    tags = data.get('tags', [])

    if not all([telegram, password, about]):
        return jsonify({"error": "Все поля должны быть заполнены"}), 400

    try:
        # Проверяем, существует ли пользователь
        existing_user = supabase.table('Stroka_test').select('*').eq('tg', telegram).execute()
        if existing_user.data:
            return jsonify({"error": "Пользователь с таким Telegram уже существует"}), 400

        # Проверяем верификацию
        if telegram not in verification_codes or not verification_codes[telegram].get('verified'):
            return jsonify({"error": "Telegram не подтвержден"}), 400

        # Извлекаем теги из описания
        prompt = f"""На основе следующего описания предложи ровно 5 наиболее релевантных тегов.
        Описание: {about}
        
        Используй только теги из следующего списка:
        {', '.join(ALL_TAGS)}
        
        Ответ должен быть в формате JSON со списком тегов, например:
        {{"tags": ["тег1", "тег2", "тег3", "тег4", "тег5"]}}
        
        Выбирай только теги из предоставленного списка."""

        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        try:
            chat_response = client.chat(
                model="mistral-tiny",
                messages=messages
            )
            
            response_text = chat_response.choices[0].message.content
            
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    suggested_tags = result.get('tags', [])
                else:
                    suggested_tags = [tag.strip() for tag in response_text.split('\n') if tag.strip()][:5]
                
                valid_tags = [tag for tag in suggested_tags if tag in ALL_TAGS]
                valid_tags = valid_tags[:5]
                while len(valid_tags) < 5:
                    for tag in ALL_TAGS:
                        if tag not in valid_tags:
                            valid_tags.append(tag)
                            if len(valid_tags) >= 5:
                                break
                
                # Объединяем теги из описания с выбранными тегами
                all_tags = list(set(valid_tags + tags))
                
                # Создаем нового пользователя
                user_data = {
                    "tg": telegram,
                    "password": password,
                    "about": about,
                    "tags": all_tags
                }
                result = supabase.table('Stroka_test').insert(user_data).execute()

                # Удаляем данные верификации
                del verification_codes[telegram]

                # Создаем JWT токен
                token = jwt.encode({
                    'user_id': result.data[0]['id'],
                    'telegram': telegram,
                    'exp': datetime.utcnow() + timedelta(days=1)
                }, JWT_SECRET, algorithm='HS256')

                return jsonify({
                    "message": "Регистрация успешна",
                    "token": token,
                    "user": {
                        "telegram": telegram,
                        "about": about,
                        "tags": all_tags
                    }
                })

            except json.JSONDecodeError:
                return jsonify({"error": "Ошибка при обработке тегов"}), 500

        except Exception as e:
            return jsonify({"error": f"Ошибка при получении тегов: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        telegram = data.get('telegram')
        password = data.get('password')

        print(f"Login attempt for telegram: {telegram}")  # Логируем попытку входа

        if not telegram or not password:
            print("Missing credentials")  # Логируем отсутствие учетных данных
            return jsonify({"error": "Не указаны учетные данные"}), 400

        # Получаем пользователя из базы данных
        result = supabase.table('Stroka_test').select('*').eq('tg', telegram).execute()
        
        print(f"Database response: {result}")  # Логируем ответ базы данных

        if not result.data:
            print(f"User not found: {telegram}")  # Логируем отсутствие пользователя
            return jsonify({"error": "Пользователь не найден"}), 404

        user = result.data[0]
        print(f"Found user: {user}")  # Логируем найденного пользователя
        
        # Проверяем пароль
        if user.get('password') != password:
            print(f"Invalid password for user: {telegram}")  # Логируем неверный пароль
            return jsonify({"error": "Неверный пароль"}), 401

        # Создаем токен
        token = jwt.encode({
            'telegram': user['tg'],
            'exp': datetime.utcnow() + timedelta(days=1)
        }, JWT_SECRET, algorithm='HS256')

        # Преобразуем теги из строки в список, если это строка
        if isinstance(user.get('tags'), str):
            try:
                user['tags'] = json.loads(user['tags'])
            except json.JSONDecodeError:
                print(f"Error parsing tags for user {telegram}")  # Логируем ошибку парсинга тегов
                user['tags'] = []

        print(f"Login successful for user: {user['tg']}")  # Логируем успешный вход

        return jsonify({
            "token": token,
            "user": {
                "telegram": user['tg'],
                "about": user.get('about', ''),
                "tags": user.get('tags', [])
            }
        })

    except Exception as e:
        print(f"Login error: {str(e)}")  # Логируем ошибку
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    return jsonify(TAGS)

@app.route('/api/suggest-tags', methods=['POST'])
def suggest_tags():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        # Если запрос пустой, возвращаем все теги
        return jsonify({"tags": ALL_TAGS})
    
    # Формируем промпт для Mistral
    prompt = f"""На основе следующего запроса предложи ровно 5 наиболее релевантных тегов для поиска.
    Запрос: {query}
    
    Используй только теги из следующего списка:
    {', '.join(ALL_TAGS)}
    
    Ответ должен быть в формате JSON со списком тегов, например:
    {{"tags": ["тег1", "тег2", "тег3", "тег4", "тег5"]}}
    
    Выбирай только теги из предоставленного списка."""

    # Отправляем запрос к Mistral
    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    try:
        chat_response = client.chat(
            model="mistral-tiny",
            messages=messages
        )
        
        # Получаем ответ от модели
        response_text = chat_response.choices[0].message.content
        
        # Извлекаем теги из ответа
        try:
            # Пытаемся найти JSON в ответе
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                suggested_tags = result.get('tags', [])
            else:
                # Если JSON не найден, разбиваем текст на строки и берем первые 5
                suggested_tags = [tag.strip() for tag in response_text.split('\n') if tag.strip()][:5]
            
            # Фильтруем теги, оставляя только те, что есть в нашем списке
            valid_tags = [tag for tag in suggested_tags if tag in ALL_TAGS]
            
            # Убеждаемся, что у нас ровно 5 тегов
            valid_tags = valid_tags[:5]
            while len(valid_tags) < 5:
                # Добавляем недостающие теги из общего списка
                for tag in ALL_TAGS:
                    if tag not in valid_tags:
                        valid_tags.append(tag)
                        if len(valid_tags) >= 5:
                            break
            
            return jsonify({"tags": valid_tags})
            
        except json.JSONDecodeError:
            # Если не удалось распарсить JSON, возвращаем пустой список
            return jsonify({"tags": []})
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"tags": []})

@app.route('/api/search-users', methods=['POST'])
def search_users():
    data = request.get_json()
    tags = data.get('tags', [])

    if not tags:
        return jsonify({"error": "Не указаны теги для поиска"}), 400

    try:
        # Получаем всех пользователей
        result = supabase.table('Stroka_test').select('*').execute()
        
        # Фильтруем результаты, чтобы показывать только пользователей с хотя бы одним совпадающим тегом
        users = {}  # Используем словарь для уникальных пользователей
        for user in result.data:
            # Преобразуем строку тегов в список, если это строка
            user_tags = user.get('tags', [])
            if isinstance(user_tags, str):
                try:
                    user_tags = json.loads(user_tags)
                except json.JSONDecodeError:
                    user_tags = []
            
            if any(tag in user_tags for tag in tags):
                # Используем telegram как ключ для уникальности
                users[user['tg']] = {
                    'telegram': user['tg'],
                    'about': user.get('about', ''),
                    'tags': user_tags
                }

        return jsonify({
            "users": list(users.values())  # Преобразуем словарь обратно в список
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/send-request', methods=['POST'])
def send_request():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            sender_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем данные запроса
        data = request.get_json()
        target_telegram = data.get('target_telegram')

        if not target_telegram:
            return jsonify({"error": "Не указан получатель заявки"}), 400

        # Проверяем, существует ли получатель
        target_user = supabase.table('Stroka_test').select('*').eq('tg', target_telegram).execute()
        if not target_user.data:
            return jsonify({"error": "Пользователь не найден"}), 404

        # Проверяем, не пытается ли пользователь отправить заявку самому себе
        if sender_telegram == target_telegram:
            return jsonify({"error": "Нельзя отправить заявку самому себе"}), 400

        # Проверяем, не отправлена ли уже заявка
        existing_outgoing = supabase.table('outgoing').select('*').eq('user_telegram', sender_telegram).eq('target_telegram', target_telegram).execute()
        if existing_outgoing.data:
            return jsonify({"error": "Заявка уже отправлена"}), 400

        # Создаем новую заявку в таблице исходящих запросов
        outgoing_data = {
            "user_telegram": sender_telegram,
            "target_telegram": target_telegram,
            "status": "pending"
        }
        outgoing_result = supabase.table('outgoing').insert(outgoing_data).execute()

        # Создаем новую заявку в таблице входящих запросов
        incoming_data = {
            "user_telegram": target_telegram,
            "sender_telegram": sender_telegram,
            "status": "pending"
        }
        incoming_result = supabase.table('incoming').insert(incoming_data).execute()

        return jsonify({
            "message": "Заявка успешно отправлена",
            "outgoing_request": outgoing_result.data[0],
            "incoming_request": incoming_result.data[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
            print(f"Getting chats for user: {user_telegram}")  # Отладочный вывод
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем входящие запросы
        try:
            incoming_requests = supabase.table('incoming').select('*').eq('user_telegram', user_telegram).eq('status', 'accepted').execute()
            print(f"Incoming requests response: {incoming_requests}")  # Отладочный вывод
        except Exception as e:
            print(f"Error fetching incoming requests: {str(e)}")  # Отладочный вывод
            raise e

        # Получаем исходящие запросы
        try:
            outgoing_requests = supabase.table('outgoing').select('*').eq('user_telegram', user_telegram).eq('status', 'accepted').execute()
            print(f"Outgoing requests response: {outgoing_requests}")  # Отладочный вывод
        except Exception as e:
            print(f"Error fetching outgoing requests: {str(e)}")  # Отладочный вывод
            raise e

        # Формируем список чатов
        chats = []
        
        # Добавляем входящие чаты
        for incoming_req in incoming_requests.data:
            chats.append({
                'id': incoming_req['id'],
                'user_telegram': incoming_req['sender_telegram'],
                'type': 'incoming',
                'status': incoming_req['status'],
                'created_at': incoming_req['created_at']
            })
        
        # Добавляем исходящие чаты
        for outgoing_req in outgoing_requests.data:
            chats.append({
                'id': outgoing_req['id'],
                'user_telegram': outgoing_req['target_telegram'],
                'type': 'outgoing',
                'status': outgoing_req['status'],
                'created_at': outgoing_req['created_at']
            })

        print(f"Returning chats: {chats}")  # Отладочный вывод
        return jsonify({
            "chats": chats
        })

    except Exception as e:
        print(f"Error in get_chats: {str(e)}")  # Отладочный вывод
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        print(f"Getting notifications for user: {user_telegram}")  # Отладочный вывод

        # Получаем входящие запросы со статусом pending
        try:
            result = supabase.table('incoming').select('*').eq('user_telegram', user_telegram).eq('status', 'pending').order('created_at', desc=True).execute()
            print(f"Incoming requests response: {result}")  # Отладочный вывод

            notifications = []
            if result.data:
                for req in result.data:
                    notifications.append({
                        'id': req['id'],
                        'type': 'incoming_request',
                        'sender_telegram': req['sender_telegram'],
                        'status': req['status'],
                        'created_at': req['created_at'],
                        'is_read': req.get('is_read', False),
                        'message': f"Новый запрос на общение от @{req['sender_telegram']}"
                    })

            print(f"Returning notifications: {notifications}")  # Отладочный вывод
            return jsonify({
                "notifications": notifications
            })

        except Exception as e:
            print(f"Error fetching incoming requests: {str(e)}")  # Отладочный вывод
            return jsonify({"error": f"Ошибка при получении входящих запросов: {str(e)}"}), 500

    except Exception as e:
        print(f"Error in get_notifications: {str(e)}")  # Отладочный вывод
        return jsonify({"error": str(e)}), 500

@app.route('/api/mark-notification-read', methods=['POST'])
def mark_notification_read():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем данные запроса
        data = request.get_json()
        notification_id = data.get('notification_id')

        if not notification_id:
            return jsonify({"error": "Не указан ID уведомления"}), 400

        # Обновляем статус уведомления
        supabase.table('incoming').update({"is_read": True}).eq('id', notification_id).execute()

        return jsonify({
            "message": "Уведомление помечено как прочитанное"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/request-verification', methods=['POST'])
def request_verification():
    data = request.get_json()
    telegram = data.get('telegram')

    if not telegram:
        return jsonify({"error": "Telegram не указан"}), 400

    try:
        # Генерируем код верификации
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
        # Сохраняем код и время его создания
        verification_codes[telegram] = {
            'code': code,
            'created_at': datetime.utcnow(),
            'verified': False
        }
        
        return jsonify({
            "message": "Код верификации создан",
            "code": code
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/check-verification', methods=['POST'])
def check_verification():
    data = request.get_json()
    telegram = data.get('telegram')
    code = data.get('code')

    if not telegram or not code:
        return jsonify({"error": "Не указан Telegram или код"}), 400

    try:
        # Проверяем существование кода
        if telegram not in verification_codes:
            return jsonify({"error": "Код верификации не найден"}), 404

        verification_data = verification_codes[telegram]
        
        # Проверяем срок действия кода (30 минут)
        if datetime.utcnow() - verification_data['created_at'] > timedelta(minutes=30):
            del verification_codes[telegram]
            return jsonify({"error": "Код верификации истек"}), 400

        # Проверяем код
        if verification_data['code'] == code:
            verification_data['verified'] = True
            return jsonify({
                "message": "Верификация успешна",
                "verified": True
            })
        else:
            return jsonify({"error": "Неверный код верификации"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/incoming-requests', methods=['GET'])
def get_incoming_requests():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем входящие запросы
        requests = supabase.table('incoming').select('*').eq('user_telegram', user_telegram).order('created_at', desc=True).execute()

        return jsonify({
            "requests": requests.data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/outgoing-requests', methods=['GET'])
def get_outgoing_requests():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем исходящие запросы
        requests = supabase.table('outgoing').select('*').eq('user_telegram', user_telegram).order('created_at', desc=True).execute()

        return jsonify({
            "requests": requests.data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/accept-request', methods=['POST'])
def accept_request():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем данные запроса
        data = request.get_json()
        request_id = data.get('request_id')

        if not request_id:
            return jsonify({"error": "Не указан ID запроса"}), 400

        # Получаем входящий запрос
        incoming_request = supabase.table('incoming').select('*').eq('id', request_id).eq('user_telegram', user_telegram).execute()
        
        if not incoming_request.data:
            return jsonify({"error": "Запрос не найден"}), 404

        request_data = incoming_request.data[0]
        
        # Обновляем статус входящего запроса
        supabase.table('incoming').update({"status": "accepted"}).eq('id', request_id).execute()
        
        # Обновляем статус соответствующего исходящего запроса
        supabase.table('outgoing').update({"status": "accepted"}).eq('user_telegram', request_data['sender_telegram']).eq('target_telegram', user_telegram).execute()

        return jsonify({
            "message": "Запрос принят"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reject-request', methods=['POST'])
def reject_request():
    try:
        # Получаем токен из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Требуется авторизация"}), 401

        token = auth_header.split(' ')[1]
        try:
            # Декодируем токен
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_telegram = decoded['telegram']
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен истек"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Недействительный токен"}), 401

        # Получаем данные запроса
        data = request.get_json()
        request_id = data.get('request_id')

        if not request_id:
            return jsonify({"error": "Не указан ID запроса"}), 400

        # Получаем входящий запрос
        incoming_request = supabase.table('incoming').select('*').eq('id', request_id).eq('user_telegram', user_telegram).execute()
        
        if not incoming_request.data:
            return jsonify({"error": "Запрос не найден"}), 404

        request_data = incoming_request.data[0]
        
        # Обновляем статус входящего запроса
        supabase.table('incoming').update({"status": "rejected"}).eq('id', request_id).execute()
        
        # Обновляем статус соответствующего исходящего запроса
        supabase.table('outgoing').update({"status": "rejected"}).eq('user_telegram', request_data['sender_telegram']).eq('target_telegram', user_telegram).execute()

        return jsonify({
            "message": "Запрос отклонен"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 