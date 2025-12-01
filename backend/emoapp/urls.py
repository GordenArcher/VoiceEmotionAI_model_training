from django.urls import path
from . import views
from rest_framework_simplejwt.views import (
    TokenRefreshView,
)

urlpatterns = [
    # Authentication routes
    path('auth/register/', views.register_view),
    path('auth/login/', views.login_view),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/logout/', views.logout_view),
    path('auth/change_password/', views.change_password),
    path('auth/delete-account/', views.delete_account_view),
    path('auth/profile/', views.get_profile_view),
    path('auth/profile/update/', views.update_profile_view),

    path('recordings/upload/', views.upload_and_analyze, name='upload-recording'),
    path('recordings/', views.list_recordings, name='list-recordings'),
    path('recordings/<int:recording_id>/', views.get_recording, name='get-recording'),
    path('recordings/<int:recording_id>/delete/', views.delete_recording, name='delete-recording'),
    path('recordings/<int:recording_id>/reanalyze/', views.reanalyze_recording, name='reanalyze-recording'),
    path('recordings/statistics/', views.user_statistics, name='user-statistics'),
    
    # Emotion Analyses
    path('analyses/', views.list_analyses, name='list-analyses'),
    path('analyses/<int:analysis_id>/', views.get_analysis, name='get-analysis'),
    
    # AI Responses
    path('ai-responses/generate/', views.generate_ai_response, name='generate-ai-response'),
    path('ai-responses/', views.list_ai_responses, name='list-ai-responses'),
    path('ai-responses/<int:response_id>/', views.get_ai_response, name='get-ai-response'),

    # Optional: health check
    path('health/', views.health_check, name='health_check'),
]
