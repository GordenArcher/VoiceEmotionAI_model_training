import numpy as np
import librosa
import pickle
from tensorflow import keras
from django.conf import settings
import os
import logging
from typing import Dict, Optional, Union, List

logger = logging.getLogger(__name__)

class EmotionRecognitionModel:
    """Singleton class to load and cache the ML model"""
    _instance = None
    _model = None
    _scaler = None
    _label_encoder = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load the trained model and preprocessing tools"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'emotion_recognition_model.h5')
            scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'scaler.pkl')
            encoder_path = os.path.join(settings.BASE_DIR, 'ml_models', 'label_encoder.pkl')

            if not os.path.exists(model_path):
                logger.error("Model file not found at %s", model_path)
                self._model = None
            else:
                self._model = keras.models.load_model(model_path)

            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self._scaler = pickle.load(f)
            else:
                logger.warning("Scaler not found at %s", scaler_path)
                self._scaler = None

            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self._label_encoder = pickle.load(f)
            else:
                logger.warning("Label encoder not found at %s", encoder_path)
                self._label_encoder = None

            if self._model is not None and self._scaler is not None and self._label_encoder is not None:
                logger.info("Emotion recognition model and artifacts loaded successfully")
            else:
                logger.warning("Emotion recognition model partially loaded or missing artifacts; predictions will be disabled until the artifacts are available.")

        except Exception as e:
            logger.exception("Unexpected error while loading model: %s", e)
            # Do not re-raise: keep service running and allow graceful degradation
            self._model = None
            self._scaler = None
            self._label_encoder = None
    
    def extract_features(self, file_path, sr=22050, duration=3):
        """
        Extract audio features from file - MUST MATCH TRAINING EXACTLY
        This extracts the same features used during model training
        """
        try:
            # Load audio with same parameters as training
            audio, sample_rate = librosa.load(
                file_path, 
                sr=sr, 
                duration=duration,
                res_type='kaiser_fast'
            )
            
            # Pad or trim to exact duration (same as training)
            target_length = sr * duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]
            
            # Extract features - EXACT SAME ORDER AND PARAMETERS AS TRAINING
            # 1. MFCCs (40 coefficients)
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            
            # 2. Chroma (12 bins)
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
            
            # 3. Mel Spectrogram (128 bins)
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            
            # 4. Spectral Contrast (7 bands)
            contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
            
            # 5. Tonnetz (6 features)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)
            
            # Combine all features in exact same order as training
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            
            # Total features: 40 + 12 + 128 + 7 + 6 = 193 features
            print(f"✓ Extracted {len(features)} features (expected: 193)")
            
            return features
        
        except Exception as e:
            print(f"✗ Error extracting features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, file_path):
        """Predict emotion from audio file"""
        # Ensure model and preprocessing artifacts are available
        if self._model is None or self._scaler is None or self._label_encoder is None:
            logger.warning("Prediction requested but model or artifacts are not loaded; returning None")
            return None

        features = self.extract_features(file_path)

        if features is None:
            return None

        # Preprocess
        try:
            features = features.reshape(1, -1)
            features_scaled = self._scaler.transform(features)

            # Predict
            prediction = self._model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_emotion = self._label_encoder.inverse_transform([predicted_class])[0]
            confidence = float(prediction[0][predicted_class] * 100)

            # Get all probabilities
            probabilities = {
                self._label_encoder.classes_[i]: float(prediction[0][i] * 100)
                for i in range(len(self._label_encoder.classes_))
            }

            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': probabilities
            }
        except Exception:
            logger.exception("Error during prediction")
            return None
