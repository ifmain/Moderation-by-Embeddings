import gradio as gr
import torch
from moderation import *  # Убедитесь, что в moderation.py есть функции getEmb и predict

# Загрузка модели
state_dict = torch.load('moderation_model.pth')
model = ModerationModel(
    embeddings_size=state_dict['embeddings_size'],
    categories_count=state_dict['categories_count'],
    hidden_layer_size=state_dict['hidden_layer_size']
).to(device)
model.load_state_dict(state_dict)

moderation.eval()  # Переключение модели в режим оценки

def predict_moderation(text):
    embeddings_for_prediction = getEmb(text)
    prediction = predict(moderation, embeddings_for_prediction)
    # Предполагая, что prediction возвращает словарь с оценками и флагом обнаружения
    category_scores = prediction.get('category_scores', {})  # Извлечение оценок категорий из словаря
    detected = prediction.get('detected', False)  # Извлечение флага обнаружения
    return category_scores, str(detected)  # Преобразование detected в строку для отображения

# Создание интерфейса Gradio
iface = gr.Interface(fn=predict_moderation,
                     inputs="text",
                     outputs=[gr.outputs.Label(label="Category Scores", type="confidences"),
                              gr.outputs.Label(label="Detected")],
                     title="Moderation Model",
                     description="Enter text to check for moderation flags.")

# Запуск интерфейса
iface.launch()