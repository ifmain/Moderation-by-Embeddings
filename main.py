from moderation import * #From files this project



# Instantiate the model with the architecture parameters
model = ModerationModel(embeddings_size=384, categories_count=11, hidden_layer_size=128).to(device)
# Load the state dict into the model
state_dict = torch.load('moderation_model_2.pth')
model.load_state_dict(state_dict)


while True:
    text=input('Text: ')
    embeddings = torch.stack(batch['embeddings']).to(device)
    scores = torch.stack(batch['scores']).to(device)
    print(json.dumps(prediction,indent=4))
    print()
