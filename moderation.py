import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_embeddings = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model_embeddings = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

embeddings_size = 384
categories_count = 11
hidden_layer_size = 128

class ModerationModel(nn.Module):
    def __init__(self, embeddings_size, categories_count, hidden_layer_size):
        super(ModerationModel, self).__init__()
        self.embeddings_size = embeddings_size
        self.categories_count = categories_count
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(self.embeddings_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.categories_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ModerationDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        categories = item['result']['categories']
        category_scores = list(categories.values())
        embeddings = getEmb(text)
        return {"embeddings": embeddings, "scores": category_scores}

def train(model, dataset, optimizer, criterion, epochs=10, batch_size=16):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        all_size=len(dataloader)
        itr=0
        for batch in dataloader:
            optimizer.zero_grad()
            
            embeddings = torch.stack(batch['embeddings']).to(device).float()
            embeddings = embeddings.transpose(0, 1)

            scores = torch.stack([torch.tensor(score, dtype=torch.float) for score in batch['scores']]).to(device)
            scores = scores.transpose(0, 1)


            outputs = model(embeddings)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch {epoch+1}, Iteration [{itr+1}/{all_size}], Loss: {loss.item()}',end='         \r')
            itr+=1
        print(f'Epoch {epoch+1}, Loss: {total_loss/all_size}',end='         \n')
    



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def getEmbeddings(sentences):
    encoded_input = tokenizer_embeddings(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model_embeddings(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.cpu()

def getEmb(text):
    sentences = [text]
    sentence_embeddings = getEmbeddings(sentences)
    return sentence_embeddings.tolist()[0]

def predict(model, embeddings):
    with torch.no_grad():
        # The embeddings are already provided as a parameter to the function, so we use them directly.
        # Ensure the embeddings are in the correct shape and device before making predictions.
        embeddings_tensor = torch.tensor(embeddings).to(device).unsqueeze(0)  # Adding batch dimension
        outputs = model(embeddings_tensor)
        predicted_scores = torch.sigmoid(outputs)
        predicted_scores = predicted_scores.squeeze(0).tolist()  # Remove batch dimension for single prediction
        category_names = ["harassment", "harassment-threatening", "hate", "hate-threatening", "self-harm", "self-harm-instructions", "self-harm-intent", "sexual", "sexual-minors", "violence", "violence-graphic"]
        
        result = {category: score for category, score in zip(category_names, predicted_scores)}
        detected = {category: score > 0.5 for category, score in zip(category_names, predicted_scores)}
        detect_value = any(value for value in detected.values())
        
        return {"category_scores": result, 'detect': detected, 'detected': detect_value}
