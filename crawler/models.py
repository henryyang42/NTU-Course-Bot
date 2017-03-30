from django.db import models


class Schedule(models.Model):
    WEEKDAY_CHOICES = (
        ('MON', '星期一'),
        ('TUE', '星期二'),
        ('WED', '星期三'),
        ('THU', '星期四'),
        ('FRI', '星期五'),
        ('SAT', '星期六'),
        ('SUN', '星期日')
    )

    TIME_CHOICES = (
        ('0', '7:00-8:00'),
        ('1', '8:10-9:00'),
        ('2', '9:10-10:00'),
        ('3', '10:20-11:10'),
        ('4', '11:20-12:10'),
        ('5', '12:20-13:10'),
        ('6', '13:20-14:10'),
        ('7', '14:20-15:10'),
        ('8', '15:30-16:20'),
        ('9', '16:30-17:20'),
        ('10', '17:30-18:20'),
        ('A', '18:30-19:20'),
        ('B', '19:25-20:15'),
        ('C', '20:25-21:15'),
        ('D', '21:20-22:10')
    )

    weekday = models.CharField(max_length=10, choices=WEEKDAY_CHOICES)
    time = models.CharField(max_length=10, choices=TIME_CHOICES)


class Department(models.Model):
    name = models.CharField(max_length=50, blank=True)
    url = models.TextField(blank=True)


class Instructor(models.Model):
    name = models.CharField(max_length=50, blank=True)
    url = models.TextField(blank=True)


class Classroom(models.Model):
    name = models.CharField(max_length=50, blank=True)
    url = models.TextField(blank=True)


class Course(models.Model):
    SEL_METHOD_CHOICES = (
        ('0', 'UNKNOWN'),
        ('1', '不限人數，直接上網加選'),
        ('2', '向教師取得授權碼後加選'),
        ('3', '有人數限制，上網登記後分發'),
    )

    semester = models.CharField(max_length=10)
    serial_no = models.CharField(max_length=10)
    # designated_for = models.ForeignKey(Department, blank=True, null=True)
    designated_for = models.CharField(max_length=10, blank=True)
    curriculum_no = models.CharField(max_length=20, blank=True)
    class_no = models.CharField(max_length=10, blank=True)
    title = models.CharField(max_length=20, blank=True)
    credits = models.CharField(max_length=10, blank=True)
    curriculum_identity_no = models.CharField(max_length=20, blank=True)
    full_half_yr = models.CharField(max_length=10, blank=True)
    required_elective = models.CharField(max_length=10, blank=True)
    # instructor = models.ForeignKey(Instructor, blank=True, null=True)
    instructor = models.CharField(max_length=10, blank=True)
    instructor_url = models.TextField(blank=True)
    sel_method = models.CharField(max_length=10, choices=SEL_METHOD_CHOICES, default='0')
    schedule = models.ManyToManyField(Schedule, blank=True)
    schedule_str = models.CharField(max_length=20, blank=True)
    # classroom = models.ForeignKey(Classroom, blank=True, null=True)
    classroom = models.CharField(max_length=20, blank=True)
    classroom_url = models.TextField(blank=True)
    capacity = models.CharField(max_length=10, blank=True)
    course_limits = models.TextField(blank=True)
    remarks = models.TextField(blank=True)

    # Syllabus related
    syllabus_url = models.TextField(blank=True)
    description = models.TextField(blank=True)
    goal = models.TextField(blank=True)
    requirements = models.TextField(blank=True)
    office_hours = models.CharField(max_length=20, blank=True)
    textbooks = models.TextField(blank=True)
    grading = models.TextField(blank=True)
    progress = models.TextField(blank=True)
    course_url = models.TextField(blank=True)

    class Meta:
        unique_together = ('semester', 'serial_no')

    def __str__(self):
        return '(%s) - %s - %s by %s' % (self.semester, self.serial_no, self.title, self.instructor)


class Review(models.Model):
    LOADING_CHOICES = (
        ('0', 'UNKNOWN'),
        ('1', '輕'),
        ('2', '尚可'),
        ('3', '中等'),
        ('4', '中上'),
        ('5', '重'),
    )
    course = models.ForeignKey(Course, blank=True, null=True)
    title = models.TextField(blank=True)
    loading = models.CharField(max_length=2, choices=LOADING_CHOICES, default='0')
    sweetness = models.CharField(max_length=2, choices=LOADING_CHOICES, default='0')
    stars = models.CharField(max_length=2, choices=LOADING_CHOICES, default='0')
    content = models.TextField(blank=True)
    sentiment = models.CharField(max_length=10, default='Neutral')
    probability = models.CharField(max_length=20, blank=True)


class User(models.Model):
    sid = models.CharField(max_length=20)
    name = models.CharField(max_length=20)
    selected_course = models.ManyToManyField(Course, blank=True)
