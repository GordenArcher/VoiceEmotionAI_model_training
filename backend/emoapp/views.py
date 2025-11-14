from django.contrib.auth import authenticate, get_user_model
from rest_framework.decorators import api_view, permission_classes, authentication_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from .models import UserProfile

User = get_user_model()

@api_view(["POST"])
@authentication_classes([])
@permission_classes([])
def register_view(request):
    data = request.data

    username = data.get("username")
    email = data.get("email")
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    password = data.get("password")
    password2 = data.get("confirm_password")
    display_name = data.get("display_name", "")

    if not all([username, email, first_name, last_name, password, password2]):
        return Response(
            {"error": "All fields are required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if password != password2:
        return Response(
            {"error": "Passwords do not match"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    if User.objects.filter(username=username).exists():
        return Response(
            {"error": "Username already taken"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    if User.objects.filter(email=email).exists():
        return Response(
            {"error": "Email already exists"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        user = User.objects.create_user(
            username=username, 
            email=email, 
            first_name=first_name, 
            last_name=last_name, 
            password=password
        )

        UserProfile.objects.create(
            user=user,
            display_name=display_name or username
        )
        
        return Response({
            "message": "Registration successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": display_name or username
            }
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response(
            {"error": "Registration failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
@permission_classes([])
@authentication_classes([])
def login_view(request):
    username = request.data.get("username")
    password = request.data.get("password")

    if not all([username, password]):
        return Response(
            {"error": "Both username and password are required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    user = authenticate(username=username, password=password)

    if not user:
        return Response(
            {"error": "Invalid credentials"}, 
            status=status.HTTP_401_UNAUTHORIZED
        )

    try:
        refresh = RefreshToken.for_user(user)
        profile = user.profile

        return Response({
            "message": "Login successful",
            "tokens": {
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            },
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "avatar": profile.avatar.url if profile.avatar else None,
                "total_recordings": profile.total_recordings
            }
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {"error": "Login failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    try:
        refresh_token = request.data.get("refresh")
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        
        return Response({
            "message": "Logout successful"
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": "Logout failed", "details": str(e)}, 
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_account_view(request):
    try:
        user = request.user
        user_data = {
            "username": user.username,
            "email": user.email
        }
        
        user.delete()
        
        return Response({
            "message": "Account deleted successfully",
            "deleted_user": user_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": "Account deletion failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_profile_view(request):
    try:
        user = request.user
        profile = user.profile

        return Response({
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "avatar": profile.avatar.url if profile.avatar else None,
                "total_recordings": profile.total_recordings,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at
            }
        }, status=status.HTTP_200_OK)
        
    except UserProfile.DoesNotExist:
        return Response(
            {"error": "User profile not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": "Failed to fetch profile", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["PUT"])
@permission_classes([IsAuthenticated])
def update_profile_view(request):
    try:
        user = request.user
        profile = user.profile
        data = request.data

        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        if 'email' in data:
            if User.objects.filter(email=data['email']).exclude(id=user.id).exists():
                return Response(
                    {"error": "Email already exists"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            user.email = data['email']
        
        user.save()

        # Update profile fields
        if 'display_name' in data:
            profile.display_name = data['display_name']
        if 'bio' in data:
            profile.bio = data['bio']
        
        profile.save()

        return Response({
            "message": "Profile updated successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "avatar": profile.avatar.url if profile.avatar else None,
                "total_recordings": profile.total_recordings
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": "Profile update failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )