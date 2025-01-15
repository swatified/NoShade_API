from django.urls import path
from .views import ToxicityAnalyzerView, home, Dashboard, Docs

urlpatterns = [
    path('analyze/', ToxicityAnalyzerView.as_view(), name='analyze'),
    path('', home.as_view(), name='home'),
    path('dashboard/', Dashboard.as_view(), name='dashboard'),
    path('docs/', Docs.as_view(), name='docs'),
]