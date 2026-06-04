import torch
import torch.nn.functional as F

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

tokenizer = DistilBertTokenizer.from_pretrained("models/text_emotion")
model = DistilBertForSequenceClassification.from_pretrained("models/text_emotion")

text = "I am very angry"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1).cpu().numpy().flatten()

print("Length:", len(probs))
print()

for i, p in enumerate(probs):
    print(f"LABEL_{i}: {p:.6f}")