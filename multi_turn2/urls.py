from django.conf.urls import url
from single_turn.views import toggle_rating
from . import views

urlpatterns = [
    url(r'^$', views.multi_turn, name='multi_turn'),
    url(r'^toggle_rating/(?P<id>\d+)$', toggle_rating, name='toggle_rating')
]
