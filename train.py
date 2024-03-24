from moderation import ModerationModel, ModerationDataset, train
import torch
import torch.optim as optim
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Передаем параметры в конструктор
model = ModerationModel(embeddings_size=384, categories_count=11, hidden_layer_size=128).to(device)

# Сохраняем модель и ее параметры
model_info = {
    'model_state_dict': model.state_dict(),
    'embeddings_size': model.embeddings_size,
    'categories_count': model.categories_count,
    'hidden_layer_size': model.hidden_layer_size
}

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Load dataset
dataset = ModerationDataset('moderation.jsonl')

# Training the model
train(model, dataset, optimizer, criterion, epochs=10)

# Save the model after training
torch.save(model_info, 'moderation_model_2.pth')
