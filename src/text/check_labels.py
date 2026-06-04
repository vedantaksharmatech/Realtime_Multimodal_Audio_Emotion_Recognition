from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    "models/text_emotion"
)

print(model.config.id2label)