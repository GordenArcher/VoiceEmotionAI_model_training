from django.contrib.auth import authenticate, get_user_model
from rest_framework.decorators import api_view, permission_classes, authentication_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from .models import UserProfile
from .helper.response import _generate_response
User = get_user_model()
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
# use User.check_password() for verifying passwords
import os
from .utils.model_loader import EmotionRecognitionModel
from .models import VoiceRecording, EmotionAnalysis, AIResponse, UserProfile
from .serializers import (
    VoiceRecordingSerializer, 
    EmotionAnalysisSerializer, 
    AIResponseSerializer
)
import json
import logging
logger = logging.getLogger(__name__)


@api_view(["POST"])
@authentication_classes([])
@permission_classes([])
def register_view(request):
    data = request.data

    print(data)

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    password2 = data.get("confirm_password")
    display_name = data.get("display_name", "")

    if not all([username, email, password, password2]):
        return Response(
            {"message": "All fields are required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if password != password2:
        return Response(
            {"message": "Passwords do not match"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    if User.objects.filter(username=username).exists():
        return Response(
            {"message": "Username already taken"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    if User.objects.filter(email=email).exists():
        return Response(
            {"message": "Email already exists"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        user = User.objects.create_user(
            username=username, 
            email=email,
            password=password
        )

        UserProfile.objects.create(
            user=user,
            display_name=display_name or username
        )
        
        refresh = RefreshToken.for_user(user)

        return Response({
            "message": "Registration successful",
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "display_name": display_name or username
            }
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response(
            {"message": "Registration failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
@permission_classes([])
@authentication_classes([])
def login_view(request):
    email = request.data.get('email')
    password = request.data.get("password")

    if not all([email, password]):
        return Response(
            {"error": "Both email and password are required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        user = get_user_model().objects.get(email=email)
    except User.DoesNotExist:
        return Response({
            "status": "error",
            "message":f"We couldn’t find an account with {email}. Please check the email or register first."
        }, status=status.HTTP_400_BAD_REQUEST) 
        

    # prefer the model's built-in check_password which handles hashing
    if not user.check_password(password):
        return Response({
            "status": "error",
            "message": "Invalid credentials. Please try again"
        }, status=status.HTTP_400_BAD_REQUEST)
        

    try:
        refresh = RefreshToken.for_user(user)
        profile = user.userprofile

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
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception:
                pass

        return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": "Logout failed", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
   data = request.data

   old_password = data.get("current_password")
   new_password = data.get("new_password")
   new_password2 = data.get("confirm_password")

   try:
       if not all([old_password, new_password, new_password2]):
           return Response({
               "status": "error",
               "message": "All fields are required"
           }, status=status.HTTP_400_BAD_REQUEST)

       if new_password != new_password2:
           return Response({
               "status": "error",
               "message": "New passwords do not match"
           }, status=status.HTTP_400_BAD_REQUEST)

       user = request.user

       if not user.check_password(old_password):
           return Response({
               "status": "error",
               "message": "Current password is incorrect"
           }, status=status.HTTP_400_BAD_REQUEST)

       if user.check_password(new_password):
           return Response({
               "status": "error",
               "message": "New password cannot be the same as the current password"
           }, status=status.HTTP_400_BAD_REQUEST)

       user.set_password(new_password)
       user.save()

       return Response({
           "status": "success",
           "message": f"Password has been changed successful"
       }, status=status.HTTP_200_OK)

   except Exception as e:
       return Response({
           "status": "error",
           "message": "Unable to change password. Please try again."
       }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



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
        profile = user.userprofile

        return Response({
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "avatar": profile.avatar.url if profile.avatar and hasattr(profile.avatar, 'url') else None,
                "total_recordings" : profile.voice_recordings.count(),
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
        profile = user.userprofile
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
                "avatar": profile.avatar.url if profile.avatar and hasattr(profile.avatar, 'url') else None,
                "total_recordings": profile.voice_recordings.count()
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": "Profile update failed", "details": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    

# ==================== VOICE RECORDING VIEWS ====================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_and_analyze(request):
    """
    Upload audio file and automatically analyze emotion
    POST /api/recordings/upload/
    """
    import json
    
    audio_file = request.FILES.get('audio_file')
    
    if not audio_file:
        return Response(
            {'error': 'No audio file provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = os.path.splitext(audio_file.name)[1].lower()
    
    if file_ext not in allowed_extensions:
        return Response(
            {'error': f'File type not supported. Allowed: {", ".join(allowed_extensions)}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate file size (10MB max)
    if audio_file.size > 10 * 1024 * 1024:
        return Response(
            {'error': 'File too large. Maximum size is 10MB'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    recording = None
    try:
        recording = VoiceRecording.objects.create(
            user=request.user.userprofile,
            audio_file=audio_file
        )
        
        # Get file path
        file_path = recording.audio_file.path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at {file_path}")
        
        print(f"\n{'='*60}")
        print(f"🎤 Processing audio: {audio_file.name}")
        print(f"📁 Saved to: {file_path}")
        print(f"📊 File size: {audio_file.size / 1024:.2f} KB")
        print(f"{'='*60}\n")
        
        model = EmotionRecognitionModel()
        prediction = model.predict(file_path)
        
        if prediction is None:
            recording.delete()
            return Response(
                {'error': 'Failed to analyze audio file - feature extraction failed'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        print(f"🎯 PREDICTION: {prediction['emotion'].upper()}")
        print(f"📈 Confidence: {prediction['confidence']:.1f}%")
        print(f"\n📊 All Probabilities:")
        sorted_probs = sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_probs:
            bar = "█" * int(prob / 2)
            print(f"   {emotion:12s} {bar:50s} {prob:5.1f}%")
        print(f"{'='*60}\n")
        
        emotion_analysis = EmotionAnalysis.objects.create(
            recording=recording,
            emotion=prediction['emotion'],
            confidence=prediction['confidence']
        )
        
        recording_serializer = VoiceRecordingSerializer(recording)
        
        return Response({
            'recording': recording_serializer.data,
            'analysis': {
                'id': emotion_analysis.id,
                'emotion': prediction['emotion'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'analyzed_at': emotion_analysis.analyzed_at
            }
        }, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        if recording and recording.pk:
            recording.delete()
        
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return Response(
            {'error': f'Error processing file: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_recordings(request):
    """
    List all recordings for authenticated user
    GET /api/recordings/
    """
    recordings = VoiceRecording.objects.filter(user=request.user.userprofile).order_by('-uploaded_at')
    serializer = VoiceRecordingSerializer(recordings, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_recording(request, recording_id):
    """
    Get specific recording
    GET /api/recordings/<id>/
    """
    recording = get_object_or_404(
        VoiceRecording, 
        id=recording_id, 
        user=request.user.userprofile
    )
    serializer = VoiceRecordingSerializer(recording)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_recording(request, recording_id):
    """
    Delete a recording
    DELETE /api/recordings/<id>/
    """
    recording = get_object_or_404(
        VoiceRecording, 
        id=recording_id, 
        user=request.user.userprofile
    )
    recording.delete()
    return Response(
        {'message': 'Recording deleted successfully'},
        status=status.HTTP_204_NO_CONTENT
    )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reanalyze_recording(request, recording_id):
    """
        Re-analyze emotion for an existing recording
        POST /api/recordings/<id>/reanalyze/
    """
    recording = get_object_or_404(
        VoiceRecording, 
        id=recording_id, 
        user=request.user.userprofile
    )
    
    try:
        model = EmotionRecognitionModel()
        prediction = model.predict(recording.audio_file.path)
        
        if prediction is None:
            return Response(
                {'error': 'Failed to analyze audio file'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        emotion_analysis = EmotionAnalysis.objects.create(
            recording=recording,
            emotion=prediction['emotion'],
            confidence=prediction['confidence']
        )
        
        return Response({
            'id': emotion_analysis.id,
            'emotion': prediction['emotion'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'analyzed_at': emotion_analysis.analyzed_at
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': f'Error analyzing file: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_statistics(request):
    """
    Get emotion statistics for authenticated user
    GET /api/recordings/statistics/
    """
    user_recordings = VoiceRecording.objects.filter(user=request.user.userprofile)
    
    # Get all emotion analyses for user
    analyses = EmotionAnalysis.objects.filter(recording__in=user_recordings)
    
    # Count emotions
    emotion_counts = {}
    total_confidence = {}
    
    for analysis in analyses:
        emotion = analysis.emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        total_confidence[emotion] = total_confidence.get(emotion, 0) + analysis.confidence
    
    emotion_stats = []
    for emotion, count in emotion_counts.items():
        emotion_stats.append({
            'emotion': emotion,
            'count': count,
            'average_confidence': total_confidence[emotion] / count,
            'percentage': (count / len(analyses) * 100) if len(analyses) > 0 else 0
        })
    
    emotion_stats.sort(key=lambda x: x['count'], reverse=True)
    
    return Response({
        'total_recordings': user_recordings.count(),
        'total_analyses': analyses.count(),
        'emotion_statistics': emotion_stats
    }, status=status.HTTP_200_OK)


# ==================== EMOTION ANALYSIS VIEWS ====================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_analyses(request):
    """
    List all emotion analyses for authenticated user
    GET /api/analyses/
    """
    user_recordings = VoiceRecording.objects.filter(user=request.user.userprofile)
    analyses = EmotionAnalysis.objects.filter(recording__in=user_recordings).order_by('-analyzed_at')
    serializer = EmotionAnalysisSerializer(analyses, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_analysis(request, analysis_id):
    """
    Get specific emotion analysis
    GET /api/analyses/<id>/
    """
    user_recordings = VoiceRecording.objects.filter(user=request.user.userprofile)
    analysis = get_object_or_404(
        EmotionAnalysis, 
        id=analysis_id, 
        recording__in=user_recordings
    )
    serializer = EmotionAnalysisSerializer(analysis)
    return Response(serializer.data, status=status.HTTP_200_OK)


# ==================== AI RESPONSE VIEWS ====================

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_ai_response(request):
    """
    Generate AI response based on emotion analysis
    POST /api/ai-responses/generate/
    Body: {"recording_id": 1}
    """
    recording_id = request.data.get('recording_id')
    
    if not recording_id:
        return Response(
            {'error': 'recording_id is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        recording = VoiceRecording.objects.get(
            id=recording_id,
            user=request.user.userprofile
        )
        
        latest_analysis = recording.emotion_analyses.order_by('-analyzed_at').first()
        
        if not latest_analysis:
            return Response(
                {'error': 'No emotion analysis found for this recording'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Generate AI response based on emotion
        response_text = _generate_response(latest_analysis.emotion, latest_analysis.confidence)
        
        # Save AI response
        ai_response = AIResponse.objects.create(
            recording=recording,
            response_text=response_text
        )
        
        serializer = AIResponseSerializer(ai_response)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    except VoiceRecording.DoesNotExist:
        return Response(
            {'error': 'Recording not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {'error': f'Error generating response: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_ai_responses(request):
    """
    List all AI responses for authenticated user
    GET /api/ai-responses/
    """
    user_recordings = VoiceRecording.objects.filter(user=request.user.userprofile)
    responses = AIResponse.objects.filter(recording__in=user_recordings).order_by('-created_at')
    serializer = AIResponseSerializer(responses, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_ai_response(request, response_id):
    """
    Get specific AI response
    GET /api/ai-responses/<id>/
    """
    user_recordings = VoiceRecording.objects.filter(user=request.user.userprofile)
    ai_response = get_object_or_404(
        AIResponse, 
        id=response_id, 
        recording__in=user_recordings
    )
    serializer = AIResponseSerializer(ai_response)
    return Response(serializer.data, status=status.HTTP_200_OK)


# ngrok http 8000/


@api_view(['GET'])
@authentication_classes([])
@permission_classes([])
def health_check(request):
    """Light-weight health check for uptime and model availability."""
    try:
        model = EmotionRecognitionModel()
        model_ready = bool(getattr(model, '_model', None) and getattr(model, '_scaler', None) and getattr(model, '_label_encoder', None))
    except Exception:
        model_ready = False

    return Response({
        'status': 'ok',
        'model_ready': model_ready
    }, status=status.HTTP_200_OK)