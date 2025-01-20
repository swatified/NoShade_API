from django import forms
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
import re

User = get_user_model()


class UserLoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Email'
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Password'
            }
        )
    )

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        password = cleaned_data.get('password')

        if email and password:
            try:
                user = User.objects.get(email=email)
                if not user.check_password(password):
                    raise ValidationError('Invalid email or password.')
                cleaned_data['user'] = user
            except User.DoesNotExist:
                raise ValidationError('Invalid email or password.')
        return cleaned_data


class UserRegistrationForm(forms.ModelForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Username'
            }
        )
    )
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Email'
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Password'
            }
        )
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full px-4 py-2 rounded-full border border-neutral-300 focus:outline-none focus:border-neutral-900',
                'placeholder': 'Confirm Password'
            }
        )
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password')

    def clean_username(self):
        username = self.cleaned_data['username']
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise ValidationError(
                'Username can only contain letters, numbers, and underscores.')
        if User.objects.filter(username=username).exists():
            raise ValidationError('This username is already taken.')
        return username

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise ValidationError('This email is already registered.')
        return email

    def clean_password(self):
        password = self.cleaned_data['password']
        if len(password) < 8:
            raise ValidationError(
                'Password must be at least 8 characters long.')
        if not re.search(r'[A-Z]', password):
            raise ValidationError(
                'Password must contain at least one uppercase letter.')
        if not re.search(r'[a-z]', password):
            raise ValidationError(
                'Password must contain at least one lowercase letter.')
        if not re.search(r'[0-9]', password):
            raise ValidationError('Password must contain at least one number.')
        return password

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            raise ValidationError('Passwords do not match.')
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
        return user
