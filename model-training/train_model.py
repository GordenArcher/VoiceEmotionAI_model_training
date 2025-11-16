"""
RAVDESS Speech Emotion Recognition - Complete Training Pipeline
Dataset: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
"""

# ==================== SETUP ====================
# Check if Drive is already mounted
import os
if os.path.exists('/content/drive/MyDrive'):
    print("Drive already mounted!")
else:
    from google.colab import drive
    drive.mount('/content/drive')

# Install required packages
# Uncomment so you can run it, this is important so you don't run into errors latr
# pip install -q librosa tensorflow scikit-learn matplotlib seaborn resampy soundfile audioread

# ==================== IMPORTS ====================
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Dataset path - Actor folders are directly in ser-dataset
RAVDESS_ROOT = "/content/drive/MyDrive/ser-dataset"

# Output folder in Google Drive
OUTPUT_FOLDER = "/content/drive/MyDrive/ser-AI-Model"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# RAVDESS emotion mapping
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("=" * 70)
print("RAVDESS SPEECH EMOTION RECOGNITION - TRAINING PIPELINE")
print("=" * 70)
print(f"Dataset: {RAVDESS_ROOT}")
print(f"Output:  {OUTPUT_FOLDER}")
print("=" * 70)

# ==================== STEP 1: DATA LOADING ====================
print("\n[1/7] Loading dataset...")

def extract_emotion_from_filename(filename):
    """Extract emotion label from RAVDESS filename"""
    parts = filename.split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, 'unknown')
    return 'unknown'

def load_ravdess_data(root_path):
    """Load all audio files and their emotion labels"""
    data = []

    for actor_folder in sorted(os.listdir(root_path)):
        actor_path = os.path.join(root_path, actor_folder)

        if os.path.isdir(actor_path) and actor_folder.startswith('Actor'):
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(actor_path, filename)
                    emotion = extract_emotion_from_filename(filename)

                    if emotion != 'unknown':
                        data.append({
                            'path': file_path,
                            'emotion': emotion,
                            'actor': actor_folder,
                            'filename': filename
                        })

    return pd.DataFrame(data)

df = load_ravdess_data(RAVDESS_ROOT)
print(f"✓ Total audio files loaded: {len(df)}")
print(f"\n📊 Emotion distribution:")
print(df['emotion'].value_counts())

# ==================== STEP 2: FEATURE EXTRACTION ====================
print("\n[2/7] Extracting audio features...")

def extract_features(file_path, sr=22050, duration=3):
    """
    Extract audio features - MUST MATCH INFERENCE CODE EXACTLY
    Total features: 193 (40 MFCCs + 12 Chroma + 128 Mel + 7 Contrast + 6 Tonnetz)
    """
    try:
        # Load audio with exact same parameters
        audio, sample_rate = librosa.load(
            file_path,
            sr=sr,
            duration=duration,
            res_type='kaiser_fast'
        )

        # Pad or trim to exact duration
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]

        # Extract features in EXACT ORDER (must match inference)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)

        # Concatenate all features
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    except Exception as e:
        print(f"✗ Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

# Extract features for all files
features_list = []
labels_list = []

print("Extracting features from audio files...")
for idx, row in df.iterrows():
    features = extract_features(row['path'])
    if features is not None:
        features_list.append(features)
        labels_list.append(row['emotion'])

    if (idx + 1) % 100 == 0:
        print(f"  Progress: {idx + 1}/{len(df)} files processed...")

X = np.array(features_list)
y = np.array(labels_list)

print(f"\n✓ Feature extraction complete!")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Expected features per sample: 193")
print(f"  Labels shape: {y.shape}")

# ==================== STEP 3: DATA PREPROCESSING ====================
print("\n[3/7] Preprocessing data...")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)

print(f"✓ Emotion classes: {list(label_encoder.classes_)}")
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

# Split data (80% train, 10% val, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Data split:")
print(f"  Training set:   {X_train_scaled.shape}")
print(f"  Validation set: {X_val_scaled.shape}")
print(f"  Test set:       {X_test_scaled.shape}")

# ==================== STEP 4: MODEL ARCHITECTURE ====================
print("\n[4/7] Building neural network model...")

def create_model(input_shape, num_classes):
    """Create a deep neural network for emotion recognition"""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),

        # First block
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second block
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Third block
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Fourth block
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

model = create_model(X_train_scaled.shape[1], y_train.shape[1])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

# ==================== STEP 5: MODEL TRAINING ====================
print("\n[5/7] Training model...")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        f'{OUTPUT_FOLDER}/best_model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("Starting training...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("✓ Training complete!")

# ==================== STEP 6: EVALUATION ====================
print("\n[6/7] Evaluating model...")

# Predictions on test set
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
test_accuracy = accuracy_score(y_test_classes, y_pred_classes)

print(f"\n{'='*70}")
print(f"TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*70}")

print("\nClassification Report:")
print(classification_report(
    y_test_classes,
    y_pred_classes,
    target_names=label_encoder.classes_
))

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix - Speech Emotion Recognition', fontsize=14, fontweight='bold')
plt.ylabel('True Emotion', fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved")
plt.show()

# ==================== STEP 7: VISUALIZATION ====================
print("\n[7/7] Creating visualizations...")

# Training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved")
plt.show()

# ==================== SAVE MODEL & ARTIFACTS ====================
print("\n" + "="*70)
print("SAVING MODEL AND ARTIFACTS")
print("="*70)

# Save model
model.save(f'{OUTPUT_FOLDER}/emotion_recognition_model.h5')
print("✓ Model saved: emotion_recognition_model.h5")

# Save label encoder
with open(f'{OUTPUT_FOLDER}/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✓ Label encoder saved: label_encoder.pkl")

# Save scaler
with open(f'{OUTPUT_FOLDER}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

# Verify scaler
print("\n Verifying scaler...")
test_sample = X_test[0].reshape(1, -1)
test_scaled = scaler.transform(test_sample)
print(f"  Original features - Min: {test_sample.min():.2f}, Max: {test_sample.max():.2f}")
print(f"  Scaled features   - Min: {test_scaled.min():.2f}, Max: {test_scaled.max():.2f}")

if test_scaled.max() > 10 or test_scaled.min() < -10:
    print("  WARNING: Scaler values are outside expected range!")
    print("  This might cause prediction issues. Consider retraining.")
else:
    print("  ✓ Scaler verification passed!")

# Save metadata
metadata = {
    'scikit_learn_version': sklearn.__version__,
    'tensorflow_version': tf.__version__,
    'librosa_version': librosa.__version__,
    'numpy_version': np.__version__,
    'n_features': int(X_train_scaled.shape[1]),
    'n_classes': int(len(label_encoder.classes_)),
    'emotions': list(label_encoder.classes_),
    'test_accuracy': float(test_accuracy),
    'train_samples': int(X_train_scaled.shape[0]),
    'val_samples': int(X_val_scaled.shape[0]),
    'test_samples': int(X_test_scaled.shape[0]),
    'total_samples': int(len(df)),
    'feature_breakdown': {
        'mfcc': 40,
        'chroma': 12,
        'mel_spectrogram': 128,
        'spectral_contrast': 7,
        'tonnetz': 6,
        'total': 193
    }
}

with open(f'{OUTPUT_FOLDER}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved: model_metadata.json")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"All files saved to: {OUTPUT_FOLDER}")
print("\nFiles created:")
print("   • emotion_recognition_model.h5  - Trained model")
print("   • scaler.pkl                    - Feature scaler")
print("   • label_encoder.pkl             - Label encoder")
print("   • model_metadata.json           - Model information")
print("   • confusion_matrix.png          - Confusion matrix visualization")
print("   • training_history.png          - Training curves")
print("="*70)
print("\nNext steps:")
print("   1. Check your Google Drive folder: ser-AI-Model")
print("   2. Download the .h5 and .pkl files")
print("   3. Copy them to your Django backend: ml_models/")
print("   4. Test with the diagnostic script")
print("="*70)