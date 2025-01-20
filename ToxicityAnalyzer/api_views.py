from rest_framework import status, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import AuthenticationFailed
from django.core.exceptions import ObjectDoesNotExist
from functools import wraps
from .serializers import (
    TextInputSerializer,
    LoginSerializer,
    RegisterSerializer,
)
from .models import ToxicityAnalyzer, APIKey, CustomUser


def track_api_usage(view_func):
    @wraps(view_func)
    def wrapper(self, request, *args, **kwargs):
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            raise AuthenticationFailed('API key is required')

        try:
            # Find the API key in database
            api_key_obj = APIKey.objects.get(key=api_key, is_active=True)
            user = api_key_obj.user

            # Check if user has subscription
            if not user.subscribed and user.api_usage >= 100:  # Free tier limit
                return Response({
                    'error': 'API usage limit reached. Please upgrade your plan.'
                }, status=status.HTTP_403_FORBIDDEN)

            # Increment API usage
            user.api_usage += 1
            user.save()

            # Add user to request
            request.user = user
            
            # Process the actual request
            return view_func(self, request, *args, **kwargs)

        except ObjectDoesNotExist:
            raise AuthenticationFailed('Invalid API key')

    return wrapper


class ToxicityAnalyzerAPIView(views.APIView):
    permission_classes = []  # Remove IsAuthenticated as we're using API key

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = ToxicityAnalyzer()

    @track_api_usage
    def post(self, request):
        serializer = TextInputSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            result = self.analyzer.analyze(text)
            return Response({
                'result': result,
                'api_calls_remaining': 100 - request.user.api_usage if not request.user.subscribed else 'unlimited'
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RegisterAPIView(views.APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': serializer.data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginAPIView(views.APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['username']
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': {
                    'username': user.username,
                    'email': user.email
                },
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutAPIView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({'message': 'Successfully logged out'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AccountDeleteAPIView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            user = request.user
            user.delete()
            return Response({'message': 'Account successfully deleted'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
