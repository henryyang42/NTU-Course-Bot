from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.single_turn, name='single_turn'),
    url(r'^toggle_rating/(?P<id>\d+)$', views.toggle_rating, name='toggle_rating')
]
