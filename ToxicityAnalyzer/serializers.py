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

class CommentAnalysisSerializer(serializers.Serializer):
    comment = serializers.CharField(required=True)
    content_id = serializers.CharField(required=True)

    def validate_comment(self, value):
        if len(value.strip()) == 0:
            raise serializers.ValidationError("Comment cannot be empty")
        return value.strip()

