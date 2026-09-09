from django.db import models


class Trial(models.Model):
    started = models.DateTimeField(auto_now_add=True)
    ended = models.DateTimeField(null=True, blank=True)
    title = models.CharField(max_length=200, blank=True)
    note = models.TextField(blank=True)
    via = models.CharField(max_length=8, blank=True)
    pack_v = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-started"]

    def __str__(self):
        return self.title or ("trial %d" % self.pk)

    @property
    def open(self):
        return self.ended is None


class Command(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name="commands")
    t_ms = models.BigIntegerField()
    text = models.CharField(max_length=240)
    source = models.CharField(max_length=8, default="ui")


class Sample(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name="samples")
    t_ms = models.BigIntegerField()
    servo = models.PositiveSmallIntegerField()
    pos = models.IntegerField(null=True)
    un = models.IntegerField(null=True)
    turns = models.IntegerField(null=True)
    speed = models.IntegerField(null=True)
    load = models.IntegerField(null=True)
    v = models.FloatField(null=True)
    temp = models.IntegerField(null=True)
    cur = models.IntegerField(null=True)
    moving = models.IntegerField(null=True)
    tq = models.CharField(max_length=4, blank=True)
    mode = models.CharField(max_length=8, blank=True)
    miss = models.BooleanField(default=False)

    class Meta:
        indexes = [models.Index(fields=["trial", "servo", "t_ms"])]


class Note(models.Model):
    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name="notes")
    t_ms = models.BigIntegerField()
    text = models.TextField()


class Limit(models.Model):
    """Host-side walls / home. Chip still clamps; this is the sitting notebook."""
    servo = models.PositiveSmallIntegerField(unique=True)
    home = models.IntegerField(null=True, blank=True)
    amin = models.IntegerField(null=True, blank=True)
    amax = models.IntegerField(null=True, blank=True)
    note = models.CharField(max_length=240, blank=True)
    updated = models.DateTimeField(auto_now=True)
