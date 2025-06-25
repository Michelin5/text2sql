# 🧠 Text2SQL AI Assistant

Интеллектуальный помощник, преобразующий естественный язык в SQL-запросы. Работает на базе FastAPI и многокомпонентной архитектуры из 11 агентов. Использует локальную базу данных на DuckDB (в виде Parquet-файлов) и систему RAG для генерации и улучшения запросов.

---

## 🚀 Возможности

- Обработка запросов на естественном языке
- Генерация SQL-запросов с использованием LLM (Gigachat API)
- Работа с реальными данными в формате Parquet
- Система агентов, разделяющая обработку на подзадачи
- Веб-интерфейс через FastAPI

---

## 📁 Структура проекта

```
text2sql/
│
├── app/                        # Основной код приложения
│   ├── agents.py
│   ├── duckdb_utils.py
│   ├── giga_wrapper.py
│   ├── main.py
│   ├── rag_handler.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env                    # переменные окружения (не пушить в git)
│   │
│   ├── chroma_db_successful_queries/
│   ├── legacy/
│   ├── sample_data/
│   └── static/
│
├── data/                       # Паркет-файлы и любые входные базы
│   ├── 1_market_access.parquet
│   ├── 2_bdmo_population.parquet
│   └── ...
│
├── docker-compose.yml          # Композиция сервисов (FastAPI + volume с parquet)
├── .dockerignore               # Исключения для docker build context
├── .gitignore                  # Исключения для Git
└── README.md                   # Описание проекта
```

---

## ⚙️ Установка и запуск

### 🐳 Вариант 1: Docker Compose (рекомендуется)

1. Убедитесь, что у вас установлен Docker и Docker Compose.
2. Поместите `.env` файл в `app/.env`.
3. Запустите:
   ```bash
   docker compose up --build
   ```
4. Приложение будет доступно на [http://localhost:5002](http://localhost:5002)

### 🐳 Вариант 2: Без Compose

```bash
docker build -t text2sql ./app
docker run -it --rm -p 5002:5002 -v $(pwd)/data:/app/data --env-file ./app/.env text2sql
```

---

## 🌐 Переменные окружения

Файл `.env` должен содержать:

```env
GIGACHAT_TOKEN=ваш_токен_от_gigachat
```

---

## 📝 Примечания

- **Parquet-файлы** — это ваша "база данных". Они подключаются при старте контейнера.
- **Volumes** используются для монтирования `data/` извне, чтобы не вшивать данные в образ.
- **Не пушьте `.env`** в публичные репозитории.
- **Приложение поднимается через FastAPI + Uvicorn**, веб-интерфейс доступен на порту 5002.

---

## 🧪 Пример запроса

После запуска откройте в браузере: [http://localhost:5002](http://localhost:5002)

Введите фразу, например:
> "Найди аномалии в популяции за 2023 год"

ИИ сгенерирует SQL-запрос, выполнит его по данным и вернёт результат.

---

## 📦 Зависимости

- Python 3.9
- FastAPI
- Uvicorn
- DuckDB
- pandas / pyarrow
- Gigachat SDK / requests

---

## 📄 Лицензия

MIT License

