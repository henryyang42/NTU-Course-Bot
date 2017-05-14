from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^toggle_rating/(?P<id>\d+)$', views.toggle_rating, name='toggle_rating'),
    url(r'^get_course/(?P<serial_no>\d+)$', views.get_course, name='get_course'),
]
