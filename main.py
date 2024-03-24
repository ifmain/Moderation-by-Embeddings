from moderation import * #From files this project


# Load model

state_dict = torch.load('moderation_model.pth')
model = ModerationModel(
    embeddings_size=state_dict['embeddings_size'],
    categories_count=state_dict['categories_count'],
    hidden_layer_size=state_dict['hidden_layer_size']
).to(device)
model.load_state_dict(state_dict)
model = model

while True:
    text=input('Text: ')
    embeddings = torch.stack(batch['embeddings']).to(device)
    scores = torch.stack(batch['scores']).to(device)
    print(json.dumps(prediction,indent=4))
    print()