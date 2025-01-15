from rest_framework import views
from rest_framework.response import Response
from .models import ToxicityAnalyzer
from .serializers import TextInputSerializer
from django.shortcuts import render

class ToxicityAnalyzerView(views.APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = ToxicityAnalyzer()

    def post(self, request):
        serializer = TextInputSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            result = self.analyzer.analyze(text)
            return Response(result)
        return Response(serializer.errors, status=400)
class home(views.APIView):
    def get(self, request):
        return render(request, 'landing.html')

class Dashboard(views.APIView):
    def get(self, request):
        return render(request, 'Dashboard.html')
class Docs(views.APIView):
    def get(self, request):
        return render(request, 'Docs.html')