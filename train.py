from moderation import ModerationModel, ModerationDataset, train
import torch
import torch.optim as optim
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Передаем параметры в конструктор
model = ModerationModel(embeddings_size=384, categories_count=11, hidden_layer_size=128).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-05)
criterion = nn.BCEWithLogitsLoss()

# Load dataset
dataset = ModerationDataset('moderation.jsonl')

# Training the model
train(model, dataset, optimizer, criterion, epochs=1)

# Correctly saving model parameters
torch.save(model.state_dict(), 'moderation_model_2.pth')
