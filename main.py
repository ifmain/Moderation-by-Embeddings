from moderation import * #From files this project

# Instantiate the model with the architecture parameters
model = ModerationModel(embeddings_size=384, categories_count=11, hidden_layer_size=128).to(device)
# Load the state dict into the model
state_dict = torch.load('moderation_model_2.pth')
model.load_state_dict(state_dict)
model.eval()

def predict_moderation(text):
    embeddings_for_prediction = getEmb(text)
    prediction = predict(model, embeddings_for_prediction)
    # Предполагая, что prediction возвращает словарь с оценками и флагом обнаружения
    category_scores = prediction.get('category_scores', {})  # Извлечение оценок категорий из словаря
    detected = prediction.get('detected', False)  # Извлечение флага обнаружения
    return category_scores, str(detected)  # Преобразование detected в строку для отображения

while True:
    text=input('Text: ')
    print(predict_moderation(text))
