#!/usr/bin/env python3
"""Launch Django + USB/BLE bridge on :8080."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
from django.core.management import execute_from_command_line

sys.argv = [sys.argv[0], "runserver", "0.0.0.0:8080"]
execute_from_command_line(sys.argv)
