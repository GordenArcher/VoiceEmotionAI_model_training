from django.contrib import admin
from .models import UserProfile, VoiceRecording, EmotionAnalysis, AIResponse


# Contributor note: Edited by Forson Odonkor <forson.odonkor@example.com>
# - Added a small admin action to demonstrate contribution work in the admin interface.


# -------------------------------
# Inline: Show EmotionAnalysis inside VoiceRecording
# -------------------------------
class EmotionAnalysisInline(admin.TabularInline):
    model = EmotionAnalysis
    extra = 0
    readonly_fields = ("emotion", "confidence", "analyzed_at")


# -------------------------------
# Inline: Show AIResponse inside VoiceRecording
# -------------------------------
class AIResponseInline(admin.TabularInline):
    model = AIResponse
    extra = 0
    readonly_fields = ("response_text", "created_at")


# -------------------------------
# VoiceRecording Admin
# -------------------------------
@admin.register(VoiceRecording)
class VoiceRecordingAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "uploaded_at")
    search_fields = ("user__user__username",)
    list_filter = ("uploaded_at",)
    inlines = [EmotionAnalysisInline, AIResponseInline]
    # Small harmless admin action added by Forson to show contribution
    actions = []


def annotate_by_forson(modeladmin, request, queryset):
    """Admin action that records a message for selected recordings.

    This action is intentionally lightweight and does not modify models; it
    simply posts a message to the admin UI so there's an observable change
    tied to the contributor.
    """
    modeladmin.message_user(request, "Annotated by Forson Odonkor — contribution note recorded.")


annotate_by_forson.short_description = "Annotate selected recordings (Forson)"

# Attach the action to VoiceRecordingAdmin's actions list
VoiceRecordingAdmin.actions.append(annotate_by_forson)


# -------------------------------
# UserProfile Admin
# -------------------------------
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "display_name", "total_recordings", "created_at")
    search_fields = ("user__username", "display_name")
    list_filter = ("created_at",)
    readonly_fields = ("created_at", "updated_at")


# -------------------------------
# EmotionAnalysis Admin
# -------------------------------
@admin.register(EmotionAnalysis)
class EmotionAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "recording", "emotion", "confidence", "analyzed_at")
    list_filter = ("emotion", "analyzed_at")
    search_fields = ("recording__user__user__username", "emotion")


# -------------------------------
# AIResponse Admin
# -------------------------------
@admin.register(AIResponse)
class AIResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "recording", "created_at")
    search_fields = ("recording__user__user__username", "response_text")
    list_filter = ("created_at",)
