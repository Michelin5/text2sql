# vis_sber.py / visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import os

def get_empty_plot_base64(message="Нет данных для визуализации"):
    import matplotlib.pyplot as plt
    import io, base64
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, fontsize=16, color='gray', ha='center', va='center')
    ax.axis('off')
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def visualize_report(df, show=False):
    if df.empty:
        if show:
            print("Нет данных для визуализации. Отправляю пустой график.")
        img_b64 = get_empty_plot_base64("Нет данных для визуализации")
        return 'ok', img_b64

    # Единственное значение (kpi)
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        if show:
            print(f"Единственное значение отчёта: {val}")
        img_b64 = get_empty_plot_base64(f"Единственное значение: {val}")
        return 'ok', img_b64

    # Только текстовые поля
    if all(df.dtypes == object):
        if show:
            print("Нет числовых данных для построения графика. Отправляю пустой график.")
        img_b64 = get_empty_plot_base64("Нет числовых данных для графика")
        return 'ok', img_b64

    # Только числовые столбцы
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        if show:
            print("В отчёте нет числовых столбцов для графика. Отправляю пустой график.")
        img_b64 = get_empty_plot_base64("Нет числовых столбцов для графика")
        return 'ok', img_b64

    img = io.BytesIO()
    fig = None

    # Bar plot
    if len(num_cols) == 1:
        fig = plt.figure(figsize=(8, 4))
        y = df[num_cols[0]]
        x = df.index if df.index.nlevels == 1 else range(len(y))
        plt.bar(x, y)
        plt.title(num_cols[0])
        plt.xlabel('Индекс')
        plt.ylabel(num_cols[0])
        plt.tight_layout()

    # Scatter
    elif len(num_cols) == 2:
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(df[num_cols[0]], df[num_cols[1]])
        plt.xlabel(num_cols[0])
        plt.ylabel(num_cols[1])
        plt.title(f"{num_cols[0]} vs {num_cols[1]}")
        plt.tight_layout()

    # Heatmap
    elif len(num_cols) > 2:
        fig = plt.figure(figsize=(min(12, len(num_cols)), 6))
        data = df[num_cols].values[:50, :10]  # ограничить размер для читабельности
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.title('Heatmap по числовым столбцам')
        plt.xlabel('Столбцы')
        plt.ylabel('Строки')
        plt.xticks(range(len(num_cols)), num_cols, rotation=45)
        plt.tight_layout()

    if fig is not None:
        plt.savefig(img, format='png')
        if show:
            plt.show()
        plt.close(fig)
        img.seek(0)
        return 'ok', base64.b64encode(img.getvalue()).decode()

    if show:
        print('Не удалось построить визуализацию для данного отчёта. Отправляю пустой график.')
    img_b64 = get_empty_plot_base64("Не удалось построить визуализацию")
    return 'ok', img_b64

if __name__ == "__main__":
    # ======== ЛОКАЛЬНЫЙ ТЕСТ НА РЕАЛЬНЫХ ДАННЫХ =========
    df = None
    # Попробуем сначала csv
    if os.path.exists("test_result.csv"):
        print("Загружаю test_result.csv")
        df = pd.read_csv("test_result.csv")
    # Если нет csv — пробуем json
    elif os.path.exists("test_result.json"):
        print("Загружаю test_result.json")
        with open("test_result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        # Если ничего нет — тестовый датасет (заглушка)
        print("Нет файлов с результатами, использую демо-данные.")
        data = {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [2, 3, 2, 3, 2]
        }
        df = pd.DataFrame(data)

    # Вызов
    status, content = visualize_report(df, show=True)
    print("Статус:", status)
    if status == "ok":
        print("Длина base64:", len(content))
        # Можно также сохранить картинку локально для проверки:
        with open("test_output.png", "wb") as f:
            f.write(base64.b64decode(content))
        print("График сохранён как test_output.png")
    else:
        print("Ошибка:", content)
