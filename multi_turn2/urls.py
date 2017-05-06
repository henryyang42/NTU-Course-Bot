from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.multi_turn, name='multi_turn'),
]
