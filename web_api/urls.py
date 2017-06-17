from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^toggle_rating/(?P<id>\d+)$', views.toggle_rating, name='toggle_rating'),
    url(r'^set_rating/(?P<group_id>\w+)/(?P<rating>[+-]*\d+)$', views.set_rating, name='set_rating'),
    url(r'^get_course/(?P<serial_no>\d+)$', views.get_course, name='get_course'),
    url(r'^tag_data$', views.tag_data, name='tag_data'),
    url(r'^tag_data_bad_rating$', views.tag_data_bad_rating, name='tag_data_bad_rating'),
    url(r'^download_log$', views.download_log, name='download_log'),
    url(r'^delete_dialogue/(?P<id>\d+)$', views.delete_dialogue, name='delete_dialogue'),
]
