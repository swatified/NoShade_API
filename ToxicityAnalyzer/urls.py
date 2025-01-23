from django.urls import path
from . import views, api_views

# Django template URLs
urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('api/generate-key/', views.APIKeyView.as_view(), name='genrate_key'),
    path('docs/', views.DocsView.as_view(), name='docs'),
    path('auth/', views.AuthView.as_view(), name='auth'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('account/delete/', views.AccountDeleteView.as_view(), name='account_delete'),
    path('profile/update/', views.ProfileUpdateView.as_view(), name='profile_update'),
    path('analytics/', views.AnalyticsView.as_view(), name = "analytics"),
]

# API endpoints
api_urlpatterns = [
    path('api/analyze/', api_views.ToxicityAnalyzerAPIView.as_view(),
         name='api_analyze'),
    path('api/register/', api_views.RegisterAPIView.as_view(), name='api_register'),
    path('api/login/', api_views.LoginAPIView.as_view(), name='api_login'),
    path('api/logout/', api_views.LogoutAPIView.as_view(), name='api_logout'),
    path('api/account/delete/', api_views.AccountDeleteAPIView.as_view(),
         name='api_account_delete'),
]

urlpatterns += api_urlpatterns
