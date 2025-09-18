from django.db import models

class Hall(models.Model):
    name = models.CharField(max_length=100)
    capacity = models.IntegerField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    floor = models.IntegerField()

    def __str__(self):
        return self.name

class Schedule(models.Model):
    hall = models.ForeignKey(Hall, on_delete=models.CASCADE)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    course_name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.course_name} in {self.hall.name}"