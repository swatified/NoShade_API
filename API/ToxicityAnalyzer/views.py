from django.shortcuts import render, redirect

import django.views.generic as generic
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.contrib.auth import get_user_model
from .forms import UserLoginForm, UserRegistrationForm
from .models import APIKey
import json



User = get_user_model()
View = generic.View
TemplateView = generic.TemplateView



class HomeView(TemplateView):
    template_name = 'landing.html'



class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
    login_url = '/auth/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        userdata = self.request.user
        context['user_data'] = {
            'first_name': self.request.user.first_name,
            'last_name': self.request.user.last_name,
            'email': self.request.user.email,
            'company': self.request.user.company,
            'subscription_play': self.request.user.subscription_plan or 'Free Plan',
            'api_usage': self.request.user.api_usage
        }
        context['api_keys'] = APIKey.objects.filter(
            user=self.request.user, is_active=True)
        return context



class APIKeyView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            key_name = data.get('name', 'Production')
            
            # Validate key name
            if not key_name or len(key_name.strip()) == 0:
                return JsonResponse({
                    'error': 'Key name is required'
                }, status=400)
                
            api_key = APIKey.objects.create(
                user=request.user,
                name=key_name
            )
            
            return JsonResponse({
                'key': api_key.key,
                'name': api_key.name,
                'created_at': api_key.created_at.strftime('%b %d, %Y')
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data'
            }, status=400)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e)
            }, status=500)


class DocsView(TemplateView):
    template_name = 'docs.html'


class LogoutView(View):
    def get(self, request):
        logout(request)
        messages.info(request, 'You have been logged out.')
        return redirect('home')


class AccountDeleteView(LoginRequiredMixin, View):
    def get(self, request):
        return render(request, 'account_delete.html')

    def post(self, request):
        user = request.user
        logout(request)
        user.delete()
        messages.success(request, 'Your account has been deleted.')
        return redirect('home')



class AuthView(View):
    template_name = 'auth.html'

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('dashboard')
        context = {
            'login_form': UserLoginForm(),
            'signup_form': UserRegistrationForm(),
            'active_form': request.GET.get('form', 'login')
        }
        return render(request, self.template_name, context)

    def post(self, request):
        form_type = request.POST.get('form_type')

        if form_type == 'login':
            form = UserLoginForm(request.POST)
            if form.is_valid():
                user = form.cleaned_data['user']
                login(request, user)
                messages.success(request, 'Successfully logged in!')
                return redirect('dashboard')
            else:
                context = {
                    'login_form': form,
                    'signup_form': UserRegistrationForm(),
                    'active_form': 'login'
                }
        else:
            form = UserRegistrationForm(request.POST)
            if form.is_valid():
                user = form.save()
                login(request, user)
                messages.success(request, 'Account created successfully!')
                return redirect('dashboard')
            context = {
                'login_form': UserLoginForm(),
                'signup_form': form,
                'active_form': 'signup'
            }

        return render(request, self.template_name, context)



class ProfileUpdateView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            data = request.POST
            user = request.user

            # Update user fields
            user.first_name = data.get('first_name', user.first_name)
            user.last_name = data.get('last_name', user.last_name)
            user.company = data.get('company', user.company)

            user.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Profile updated successfully',
                'data': {
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'company': user.company,
                }
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)



class SubscriptionView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            # Subscription handling logic
            pass
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
