from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/text_emotion",
    tokenizer="models/text_emotion"
)

tests = [
    "I am very angry",
    "I am scared",
    "I am extremely happy today",
    "I feel sad"
]

for t in tests:
    print("\nTEXT:", t)
    print(classifier(t))