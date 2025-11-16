# Speech Emotion Recognition - Model Training

Train the emotion recognition model on RAVDESS dataset.

## Setup

1. Upload RAVDESS dataset to Google Drive
2. Open Google Colab
3. Run `train_model.py` in Colab
4. Download trained model files from Drive

## Output Files

- `emotion_recognition_model.h5` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `model_metadata.json` - Model info

## Copy to Backend

After training, copy the `.h5` and `.pkl` files to:
```
backend/ml_models/
```

## Dataset

RAVDESS - 8 emotions from 24 actors
- neutral, calm, happy, sad, angry, fearful, disgust, surprised