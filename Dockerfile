FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование только requirements.txt сначала
COPY requirements.txt .

# Обновление pip и базовых инструментов
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Установка базовых зависимостей
RUN pip install --no-cache-dir \
    Flask==3.0.2 \
    flask-cors==4.0.0 \
    Werkzeug==3.0.1 \
    click==8.1.7 \
    itsdangerous==2.1.2 \
    Jinja2==3.1.3 \
    MarkupSafe==2.1.5

# Установка дополнительных зависимостей
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    httpx==0.27.0 \
    PyJWT==2.8.0 \
    python-dotenv==1.0.0 \
    gunicorn==21.2.0

# Установка специфических зависимостей
RUN pip install --no-cache-dir \
    mistralai==0.0.12 \
    supabase==2.3.5

# Копирование остальных файлов проекта
COPY . .

# Открытие порта
EXPOSE 5000

# Запуск приложения через gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"] 