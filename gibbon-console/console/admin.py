from django.contrib import admin
from .models import Command, Note, Sample, Trial


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "started", "ended", "via", "pack_v")


@admin.register(Command)
class CommandAdmin(admin.ModelAdmin):
    list_display = ("id", "trial", "t_ms", "text")


@admin.register(Sample)
class SampleAdmin(admin.ModelAdmin):
    list_display = ("id", "trial", "t_ms", "servo", "pos", "load", "v")


@admin.register(Note)
class NoteAdmin(admin.ModelAdmin):
    list_display = ("id", "trial", "t_ms", "text")
