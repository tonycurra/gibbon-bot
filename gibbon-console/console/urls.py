from django.urls import path
from . import views

urlpatterns = [
    path("", views.live, name="live"),
    path("graphs/", views.graphs, name="graphs"),
    path("act/", views.act, name="act"),
    path("trials/", views.trials, name="trials"),
    path("trials/<int:pk>/", views.trial_detail, name="trial"),
    path("api/status", views.api_status),
    path("api/rx", views.api_rx),
    path("api/tx", views.api_tx),
    path("api/trial", views.api_trial),
    path("api/trial/stop", views.api_trial_stop),
    path("api/trial/note", views.api_trial_note),
    path("api/trials/<int:pk>/samples", views.api_samples),
]
