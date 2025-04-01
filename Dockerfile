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

# Обновление pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Установка зависимостей по одной
RUN pip install --no-cache-dir Flask==3.0.2
RUN pip install --no-cache-dir flask-cors==4.0.0
RUN pip install --no-cache-dir supabase==1.2.0
RUN pip install --no-cache-dir PyJWT==2.8.0
RUN pip install --no-cache-dir python-dotenv==1.0.0
RUN pip install --no-cache-dir gunicorn==21.2.0
RUN pip install --no-cache-dir requests==2.31.0
RUN pip install --no-cache-dir mistralai==0.0.12

# Копирование остальных файлов проекта
COPY . .

# Открытие порта
EXPOSE 5000

# Запуск приложения через gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]