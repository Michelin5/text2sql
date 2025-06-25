Как запустить фронт:

1. Создать файл .env с GIGACHAT_TOKEN=<токен>
2. Запустить файл main.py
3. Открыть index.html в браузере

_______________________________________
Как пользоваться giga_wrapper.py:

```python
from giga_wrapper import call_giga_api_wrapper

call_giga_api_wrapper("сообщение", "системный промпт")
```
_______________________________________
Агенты расположены в agents.py, папку legacy можете игнорить

