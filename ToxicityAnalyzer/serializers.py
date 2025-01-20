from rest_framework import serializers

class TextInputSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=5000)

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=100)
    password = serializers.CharField(max_length=100)
class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=100)
    password = serializers.CharField(max_length=100)
    email = serializers.CharField(max_length=100)

class LogoutSerializer(serializers.Serializer):
    pass

