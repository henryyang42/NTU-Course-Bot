from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.single_turn, name='single_turn'),
]
