from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'', views.user_simulator, name='user_simulator'),
]
