from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("console", "0002_limit"),
    ]

    operations = [
        migrations.CreateModel(
            name="Calibration",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("board", models.CharField(max_length=32)),
                ("fw", models.CharField(blank=True, max_length=16)),
                ("via", models.CharField(blank=True, max_length=8)),
                ("joints", models.JSONField(default=dict)),
            ],
            options={
                "ordering": ["-created"],
            },
        ),
    ]
