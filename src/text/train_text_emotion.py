# src/text/train_text_emotion.py

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

print("=" * 60)
print("Loading GoEmotions Dataset")
print("=" * 60)

# --------------------------------------------------
# Download GoEmotions automatically
# --------------------------------------------------
dataset = load_dataset("google-research-datasets/go_emotions")

df = dataset["train"].to_pandas()

print("Columns:", df.columns.tolist())
print("Total samples:", len(df))

# --------------------------------------------------
# Remove empty labels
# --------------------------------------------------
df = df[df["labels"].map(len) > 0]

# Convert multi-label -> single label
# Keep first label only
df["label"] = df["labels"].apply(lambda x: x[0])

df = df[["text", "label"]]

num_labels = df["label"].nunique()

print("Number of classes:", num_labels)

print("\nClass Distribution:")
print(df["label"].value_counts().sort_index())

# --------------------------------------------------
# Stratified Train / Validation Split
# --------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df["label"]
)

print("\nTrain Samples:", len(train_df))
print("Validation Samples:", len(val_df))

# --------------------------------------------------
# HuggingFace Dataset
# --------------------------------------------------
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------
print("\nLoading DistilBERT tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# --------------------------------------------------
# Model
# --------------------------------------------------
print("\nLoading DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }

# --------------------------------------------------
# Training Arguments
# --------------------------------------------------
training_args = TrainingArguments(

    output_dir="models/text_emotion",

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    save_total_limit=2,

    logging_steps=100,

    report_to="none"
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --------------------------------------------------
# Train
# --------------------------------------------------
print("\nStarting DistilBERT Training...\n")

trainer.train()

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
print("\nRunning Final Evaluation...\n")

metrics = trainer.evaluate()

print(
    "Validation Accuracy:",
    round(metrics["eval_accuracy"] * 100, 2),
    "%"
)

# --------------------------------------------------
# Save Model
# --------------------------------------------------
print("\nSaving model...")

trainer.save_model("models/text_emotion")
tokenizer.save_pretrained("models/text_emotion")

print("\nTraining Complete")
print("Model saved to: models/text_emotion")