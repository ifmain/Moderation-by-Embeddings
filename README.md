# moderation by embeddings

This is a simple multilingual model for text moderation using embeddings.

PS: Although this model itself is MIT, it uses sentence `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` under `license: apache-2.0`.

exaple usage:

```python
from moderation import * #From files this project

# Load model
moderation = ModerationModel()
moderation.load_state_dict(torch.load('moderation_model.pth'))

# Test text
text = "I want to kill them."

embeddings_for_prediction = getEmb(text)
prediction = predict(moderation, embeddings_for_prediction)
print(json.dumps(prediction,indent=4))
```

Output:
```json
{
    "category_scores": {
        "harassment": 0.039179909974336624,
        "harassment-threatening": 0.5689294338226318,
        "hate": 0.0096114631742239,
        "hate-threatening": 0.00895680021494627,
        "self-harm": 0.0008832099265418947,
        "self-harm-instructions": 2.1136918803676963e-05,
        "self-harm-intent": 0.00033596932189539075,
        "sexual": 5.425313793239184e-05,
        "sexual-minors": 5.160131422599079e-06,
        "violence": 0.9684166312217712,
        "violence-graphic": 0.0015151903498917818
    },
    "detect": {
        "harassment": false,
        "harassment-threatening": true,
        "hate": false,
        "hate-threatening": false,
        "self-harm": false,
        "self-harm-instructions": false,
        "self-harm-intent": false,
        "sexual": false,
        "sexual-minors": false,
        "violence": true,
        "violence-graphic": false
    },
    "detected": true
}
```


This model covert embedings to moderaton score
The dataset helped with normalizing the model output, but the model does not include rows from the dataset

HuggingFace: !(ifmain/moderation_by_embeddings)[https://huggingface.co/ifmain/moderation_by_embeddings]
