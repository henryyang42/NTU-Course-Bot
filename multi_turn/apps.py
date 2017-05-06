from django.apps import AppConfig
from django.conf import settings


class MultiTurnConfig(AppConfig):
    name = 'multi_turn'

    # Run only once on start.
    def ready(self):

        if not settings.DEBUG:  # Only load model in production to speed up debugging.

            print('[Info] Multi-turn LU model loaded.')

        else:
            print('[Info] Under DEBUG mode, multi-turn LU is not loaded.')
