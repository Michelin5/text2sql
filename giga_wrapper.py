from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

# --- Настройка GigaChat ---
GIGA_CREDENTIALS = os.environ.get("GIGACHAT_TOKEN")
giga_client = None
try:
    giga_client = GigaChat(credentials=GIGA_CREDENTIALS, verify_ssl_certs=False, model="GigaChat-2-Max")
except Exception as e:
    print(f"[ОШИБКА GigaChat Init] {e}")




def call_giga_api_wrapper(prompt_text, system_instruction):
    if giga_client is None: return "[ОШИБКА API] Клиент GigaChat не инициализирован."
    print(f"\nВызов GigaChat: Системная инструкция (начало): {system_instruction[:100]}... Промпт: {prompt_text[:100]}...")
    try:
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt_text)]
        res = giga_client.invoke(messages)
        print(f"GigaChat ответ (начало): {res.content[:100]}...")
        return res.content.strip()
    except Exception as e:
        print(f"Ошибка вызова GigaChat API: {e}")
        return f"[ОШИБКА API] {e}"