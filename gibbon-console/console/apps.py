import os

from django.apps import AppConfig


class ConsoleConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "console"

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return
        from django.db.backends.signals import connection_created
        from . import bridge
        connection_created.connect(_sqlite_pragma)
        bridge.start()


def _sqlite_pragma(sender, connection, **kwargs):
    if connection.vendor != "sqlite":
        return
    cur = connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA busy_timeout=8000;")
    cur.execute("PRAGMA synchronous=NORMAL;")
