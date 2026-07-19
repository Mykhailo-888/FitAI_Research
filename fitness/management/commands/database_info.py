from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Print auditable database backend and PostgreSQL server information."

    def handle(self, *args, **options):
        settings = connection.settings_dict
        self.stdout.write(f"Django database engine: {settings['ENGINE']}")
        self.stdout.write(f"Database host: {settings.get('HOST') or '(local file)'}")
        self.stdout.write(f"Database name: {settings.get('NAME')}")
        if connection.vendor != "postgresql":
            self.stdout.write("PostgreSQL server version: NOT IN USE")
            return
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version")
            version = cursor.fetchone()[0]
        self.stdout.write(f"PostgreSQL server version: {version}")
