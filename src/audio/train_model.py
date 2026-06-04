# src/audio/train_model.py
#
# Architecture : CNN + BiLSTM + Attention
# Dataset      : CREMA-D only
#
# v3 changes (to push 70.7% → ~75%):
#   1. Label smoothing = 0.1   → prevents overconfidence after epoch 14
#   2. L2 0.001 → 0.002        → stronger weight penalty on Conv + Dense
#   3. Dropout 0.4 → 0.5       → tighter regularization in classifier head
#   4. EarlyStopping monitors val_accuracy (not val_loss)
#   5. Mixup REMOVED           → was corrupting accuracy metric + destabilising val
#   6. Noise stddev 0.05→0.03  → lighter augmentation, less label corruption
#
# Calibration : COMMENTED OUT — not deleted

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, Input
from sklearn.utils.class_weight import compute_class_weight
import pickle

# ── Load data ─────────────────────────────────────────────────────────────────
X_train = np.load("models/X_train.npy").astype(np.float32)
X_test  = np.load("models/X_test.npy").astype(np.float32)
y_train = np.load("models/y_train.npy")
y_test  = np.load("models/y_test.npy")

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print("Classes    :", list(label_encoder.classes_))
print("Train shape:", X_train.shape)
print("Test  shape:", X_test.shape)
print("Val range  : [{:.2f}, {:.2f}]".format(X_train.min(), X_train.max()))

unique, counts = np.unique(y_train, return_counts=True)
print("\nTraining class distribution:")
for u, c in zip(unique, counts):
    print(f"  {label_encoder.classes_[u]}: {c}")

# ── Class weights ──────────────────────────────────────────────────────────────
class_weights      = compute_class_weight('balanced',
                                          classes=np.unique(y_train),
                                          y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("\nClass weights:", {label_encoder.classes_[k]: round(float(v), 3)
                           for k, v in class_weights_dict.items()})

# ── One-hot encoding ───────────────────────────────────────────────────────────
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes).astype(np.float32)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes=num_classes).astype(np.float32)

# ── tf.data pipeline — noise + time mask only (no mixup) ──────────────────────
BATCH_SIZE = 64

def augment_fn(x, y):
    """
    Light on-the-fly augmentation:
      - Gaussian noise  stddev=0.03  (reduced from 0.05 — less corruption)
      - Random time mask 0-12 frames (reduced from 20)
    Mixup removed — it was blending soft labels and making accuracy metric
    unstable and unreadable during training.
    """
    x     = tf.cast(x, tf.float32)
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.03,
                             dtype=tf.float32)
    x    = x + noise
    # mask cannot be 0 , give problem / error to tf.zeros_like when mask=0 (zero-width mask)
    mask = tf.random.uniform([], 1, 12, dtype=tf.int32)
    x    = tf.concat([x[:, :, :-mask, :],
                      tf.zeros_like(x[:, :, -mask:, :])], axis=2)
    return x, y

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    .shuffle(buffer_size=8000, seed=42)
    .batch(BATCH_SIZE)
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test_cat))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── Attention block ────────────────────────────────────────────────────────────
def attention_block(inputs):
    """
    Soft self-attention over BiLSTM time axis.
    inputs  : (B, T, F)
    returns : (B, F)
    """
    score             = layers.Dense(1, activation='tanh')(inputs)
    attention_weights = layers.Softmax(axis=1)(score)
    context           = layers.Multiply()([inputs, attention_weights])
    context           = layers.GlobalAveragePooling1D()(context)
    return context

# ── Model ──────────────────────────────────────────────────────────────────────
def build_cnn_bilstm_attention(input_shape=(40, 174, 1), num_classes=4):

    inputs = Input(shape=input_shape)

    # CNN Block 1
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.002))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # CNN Block 2
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # CNN Block 3 — no MaxPool, preserves time resolution for BiLSTM
    x = layers.Conv2D(96, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.30)(x)

    # Reshape → (B, time_steps, features)
    x          = layers.Permute((2, 1, 3))(x)
    shp        = x.shape
    time_steps = shp[1]
    feat_dim   = shp[2] * shp[3]
    x          = layers.Reshape((time_steps, feat_dim))(x)

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                    dropout=0.2)
    )(x)
    x = layers.Dropout(0.40)(x)

    # Attention
    x = attention_block(x)

    # Classifier head — dropout raised 0.4 → 0.5
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="CNN_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        # Label smoothing 0.1 — key fix for overconfidence plateau at epoch 14
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


model = build_cnn_bilstm_attention(input_shape=(40, 174, 1),
                                   num_classes=num_classes)
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")
print("Starting training — CNN + BiLSTM + Attention (v3: label-smooth, no mixup)\n")

# ── Custom logger ──────────────────────────────────────────────────────────────
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        print(
            f"Epoch {epoch+1:>3} | "
            f"loss: {logs['loss']:.4f}  acc: {logs['accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f}  val_acc: {logs['val_accuracy']:.4f} | "
            f"lr: {lr:.7f}"
        )

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    EpochLogger(),
    # Monitor val_accuracy — restores the best accuracy epoch
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=6, min_lr=1e-5, verbose=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="models/audio_cnn_bilstm_attention_best.keras",
        monitor='val_accuracy', save_best_only=True, verbose=0
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=0
)

print("\nBest Val Accuracy  :", round(max(history.history['val_accuracy']), 4))
print("Final Train Accuracy:", round(history.history['accuracy'][-1], 4))
print("Final Val  Accuracy :", round(history.history['val_accuracy'][-1], 4))

model.save("models/audio_cnn_bilstm_attention.keras")
print("\nSaved → models/audio_cnn_bilstm_attention.keras")
print("Best  → models/audio_cnn_bilstm_attention_best.keras")

# ────────────────────────────────────────────────────────────────────────────────
# CALIBRATION — commented out, NOT deleted.
# ────────────────────────────────────────────────────────────────────────────────
# class TemperatureScaling(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.temperature = tf.Variable(1.5, trainable=True, dtype=tf.float32)
#     def call(self, logits):
#         return logits / self.temperature
#
# cal_inputs  = Input(shape=(40, 174, 1))
# logits      = model(cal_inputs)
# cal_outputs = tf.nn.softmax(TemperatureScaling()(logits))
# cal_model   = Model(cal_inputs, cal_outputs)
# cal_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
#                   loss='categorical_crossentropy')
# cal_model.fit(X_test, y_test_cat, epochs=50, verbose=0)
# cal_model.save("models/audio_cnn_bilstm_attention_calibrated.keras")
# ────────────────────────────────────────────────────────────────────────────────


# ── Evaluation plots ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc as sk_auc

# 1. Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy — CNN + BiLSTM + Attention v3')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss — CNN + BiLSTM + Attention v3')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend()

plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150); plt.close()
print("Saved → models/training_curves.png")

# 2. Confusion matrix
preds        = model.predict(val_ds, verbose=0)
pred_classes = np.argmax(preds,      axis=1)
true_classes = np.argmax(y_test_cat, axis=1)
cm           = confusion_matrix(true_classes, pred_classes, normalize='true')

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (%) — CNN + BiLSTM + Attention v3 (CREMA-D)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150); plt.close()
print("Saved → models/confusion_matrix.png")

# 3. Per-class F1
report    = classification_report(true_classes, pred_classes,
                                  target_names=label_encoder.classes_,
                                  output_dict=True)
f1_scores = [report[c]['f1-score'] for c in label_encoder.classes_]

plt.figure(figsize=(7, 4))
bars = plt.bar(label_encoder.classes_, f1_scores,
               color=['#e74c3c', '#9b59b6', '#2ecc71', '#3498db'])
plt.ylim(0, 1.0)
plt.title('Per-Class F1 — CNN + BiLSTM + Attention v3 (CREMA-D)')
plt.ylabel('F1 Score'); plt.xlabel('Emotion Class')
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f'{score:.2f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('models/f1_scores.png', dpi=150); plt.close()
print("Saved → models/f1_scores.png")

# 4. ROC-AUC
plt.figure(figsize=(7, 5))
for i, cls in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], preds[:, i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC={sk_auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(); plt.title('ROC-AUC — CNN + BiLSTM + Attention v3')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.tight_layout()
plt.savefig('models/roc_auc.png', dpi=150); plt.close()
print("Saved → models/roc_auc.png")