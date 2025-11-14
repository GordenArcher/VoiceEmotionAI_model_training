from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    display_name = models.CharField(max_length=100, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    avatar = models.ImageField(upload_to="avatars/", blank=True, null=True)
    total_recordings = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.display_name or self.user.username


class VoiceRecording(models.Model):
    """
    Stores user voice recordings
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="voice_recordings")
    audio_file = models.FileField(upload_to="voice_recordings/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.uploaded_at}"


class EmotionAnalysis(models.Model):
    """
        Stores results of emotion analysis
    """
    recording = models.ForeignKey(VoiceRecording, on_delete=models.CASCADE, related_name="emotion_analyses")
    emotion = models.CharField(max_length=50) 
    confidence = models.FloatField() 
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.recording.user.username} - {self.emotion} ({self.confidence})"


class AIResponse(models.Model):
    """
        Stores AI-generated text responses related to a recording
    """
    recording = models.ForeignKey(VoiceRecording, on_delete=models.CASCADE, related_name="ai_responses")
    response_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.recording.user.username} - Response at {self.created_at}"
