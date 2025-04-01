FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование только requirements.txt сначала
COPY requirements.txt .

# Обновление pip и установка зависимостей
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Установка всех зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остальных файлов проекта
COPY . .

# Открытие порта
EXPOSE 5000

# Запуск приложения через gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]