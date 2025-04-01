# STROKA

Веб-приложение для поиска собеседников по интересам.

## Функциональность

- Регистрация и авторизация пользователей через Telegram
- Поиск собеседников по тегам интересов
- Система запросов на общение
- Чат между пользователями
- Уведомления о новых запросах

## Технологии

- Backend: Python (Flask)
- Database: Supabase
- Frontend: HTML, CSS, JavaScript
- API: Mistral AI для анализа интересов

## Установка и запуск локально

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Artem336600/STROKA.git
cd STROKA
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Создайте файл .env и добавьте необходимые переменные окружения:
```
SUPABASE_URL=ваш_url
SUPABASE_KEY=ваш_ключ
JWT_SECRET=ваш_секретный_ключ
MISTRAL_API_KEY=ваш_ключ_mistral
```

4. Запустите сервер:
```bash
python server.py
```

## Деплой на Railway

1. Форкните этот репозиторий на GitHub
2. Создайте новый проект на Railway, выбрав "Deploy from GitHub repo"
3. Выберите ваш форк репозитория
4. Добавьте переменные окружения в настройках проекта:
   - SUPABASE_URL
   - SUPABASE_KEY
   - JWT_SECRET
   - MISTRAL_API_KEY
5. Railway автоматически запустит сервер, используя конфигурацию из Procfile

## Структура проекта

- `server.py` - основной файл сервера
- `static/` - статические файлы (CSS, JavaScript)
- `templates/` - HTML шаблоны
- `requirements.txt` - зависимости проекта
- `Procfile` - конфигурация для деплоя
