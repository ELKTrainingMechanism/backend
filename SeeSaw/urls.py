from django.contrib import admin
from django.urls import path
from frontend import views
from django.views.decorators.csrf import ensure_csrf_cookie

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/get_csrf_token/', ensure_csrf_cookie(views.get_csrf_token), name='get_csrf_token'),
    path('api/post-data/', views.post_data, name='post_data'),
]

