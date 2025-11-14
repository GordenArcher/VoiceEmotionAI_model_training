from django.urls import path
from . import views

urlpatterns = [
    # Authentication routes
    path('auth/register/', views.register_view),
    path('auth/login/', views.login_view),
    path('auth/logout/', views.logout_view),
    path('auth/delete-account/', views.delete_account_view),
    path('auth/profile/', views.get_profile_view),
    path('auth/profile/update/', views.update_profile_view),

    # Optional: health check
    # path('health/', views.health_check, name='health_check'),
]
