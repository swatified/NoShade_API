from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ToxicityAnalyzer.urls')),
    path('', include('ToxicityAnalyzer.urls')),
]