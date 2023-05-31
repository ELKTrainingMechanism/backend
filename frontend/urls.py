from django.urls import path
from . import views

urlpatterns = [
    path('endpoint/', views.MyEndpoint.as_view(), name='my_endpoint'),
    path('get_csrf_token/', views.get_csrf_token)
    # Additional API endpoints for your app
]

