from django.urls import path
from .views import recognize_hall

urlpatterns = [
    path('api/recognize_hall/', recognize_hall, name='recognize_hall'),
]