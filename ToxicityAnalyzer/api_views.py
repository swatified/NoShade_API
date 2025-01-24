# We'll need to create Comment and Content models
from .models import ToxicityAnalyzer, Comment, Content, CustomUser, APIKey
from .serializers import CommentAnalysisSerializer
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import AuthenticationFailed
from django.core.exceptions import ObjectDoesNotExist
from functools import wraps
from .serializers import (
    TextInputSerializer,
    LoginSerializer,
    RegisterSerializer,
)
from django.contrib.auth import get_user_model


def getUserByToken(token):
    accesstoken = AccessToken(token)
    user_id = accesstoken['user_id']
    return user_id


def check_token(view_func):
    @wraps(view_func)
    def wrapper(self, request, *args, **kwargs):
        token = request.headers.get('X-AUTH-TOKEN')
        refresh = request.headers.get('X-REFRESH-TOKEN')
        try:
            AccessToken(token)
            return view_func(self, request, *args, **kwargs)
        except Exception as e:
            userId = getUserByToken(token)
            refresh = RefreshToken.for_user(userId)
            newToken = refresh.token
            print(refresh)
    return wrapper


def track_api_usage(view_func):
    @wraps(view_func)
    def wrapper(self, request, *args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            raise AuthenticationFailed('API key is required')

        try:
            api_key_obj = APIKey.objects.get(key=api_key, is_active=True)
            user = api_key_obj.user

            if not user.subscribed and user.api_usage >= 100:
                return Response({
                    'error': 'API usage limit reached. Please upgrade your plan.'
                }, status=status.HTTP_403_FORBIDDEN)

            user.api_usage += 1
            user.save()

            request.user = user
            return view_func(self, request, *args, **kwargs)

        except ObjectDoesNotExist:
            raise AuthenticationFailed('Invalid API key')

    return wrapper


class ToxicityAnalyzerAPIView(views.APIView):
    permission_classes = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = ToxicityAnalyzer()

    def get_or_create_content(self, content_id, user):
        content, created = Content.objects.get_or_create(
            external_id=content_id,
            defaults={'created_by': user}
        )
        return content, created

    @track_api_usage
    def post(self, request):
        serializer = CommentAnalysisSerializer(data=request.data)
        api_key = request.headers.get('X-API-Key')
        api_key_obj = APIKey.objects.get(key=api_key, is_active=True)
        user = api_key_obj.user
        print(user)

        if serializer.is_valid():
            comment_text = serializer.validated_data['comment']
            content_id = serializer.validated_data['content_id']

            try:
                content, created = self.get_or_create_content(content_id, user)

                analysis_result = self.analyzer.analyze(comment_text)
                print("Analysis result:", analysis_result)  # For debugging

                # Directly access the fields from the response
                sentiment = analysis_result['sentiment']
                toxic_labels = [word for line in analysis_result['toxic_labels'].split('\n') for word in line.split()]
                print(toxic_labels)
                is_toxic = bool(analysis_result['toxic_labels'].strip())

                # Create comment with correct field mapping
                comment = Comment.objects.create(
                    content=content,
                    text=comment_text,
                    sentiment=sentiment,  # Direct access
                    toxic_labels=toxic_labels,  # Direct access
                    is_toxic=is_toxic,
                    analyzed_by=request.user
                )

                # Prepare response with correct field mapping
                response_data = {
                    'comment_id': comment.id,
                    'content_id': content_id,
                    'content_status': 'created' if created else 'existing',
                    'analysis': {
                        'sentiment': sentiment,  # Direct access
                        'toxic_labels': toxic_labels,
                        'is_toxic': is_toxic,
                        'recommendation': 'reject' if is_toxic else 'approve'
                    },
                    'api_calls_remaining': 'unlimited' if request.user.subscribed else 100 - request.user.api_usage
                }
                return Response(response_data, status=status.HTTP_200_OK)

            except Exception as e:
                print(e)
                # Add proper logging here
                return Response({
                    'error': 'Error processing comment',
                    'detail': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RegisterAPIView(views.APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            User = get_user_model()
            user = User.objects.create_user(
                username=serializer.validated_data['username'],
                email=serializer.validated_data['email'],
                password=serializer.validated_data['password']
            )
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
            username = serializer.validated_data['username']
            User = get_user_model()

            try:
                user = User.objects.get(username=username)
                refresh = RefreshToken.for_user(user)
                return Response({
                    'user': {
                        'username': user.username,
                        'email': user.email
                    },
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                })
            except User.DoesNotExist:
                return Response(
                    {'error': 'User not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutAPIView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            if not refresh_token:
                return Response(
                    {'error': 'Refresh token is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response(
                {'message': 'Successfully logged out'},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


class AccountDeleteAPIView(views.APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            user = request.user
            user.delete()
            return Response(
                {'message': 'Account successfully deleted'},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
