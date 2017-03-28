import django
import os
import re
from bs4 import BeautifulSoup
from crawler.const import base_url

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()

from crawler.models import *


def generate_rows(soup):
    rows = soup.find_all('tr')
    row_dicts = []
    for row in rows:
        tds = row.find_all('td')
        row_dicts.append({
            'serial_no': tds[0].get_text(),
            'designated_for': tds[1].get_text(),
            'curriculum_no': tds[2].get_text(),
            'class_no': tds[3].get_text(),
            'title': tds[4].get_text(),
            'credits': tds[5].get_text(),
            'curriculum_identity_no': tds[6].get_text(),
            'full_half_yr': tds[7].get_text(),
            'required_elective': tds[8].get_text(),
            'instructor': tds[9].get_text(),
            'sel_method': tds[10].get_text(),
            'schedule_str': re.sub(r'\([^)]*\)', '', tds[11].get_text()),
            'capacity': tds[12].get_text(),
            'course_limits': tds[13].get_text(),
            'remarks': tds[14].get_text(),
            'syllabus_url': base_url + tds[4].a['href'] if tds[4].a else '',
            'instructor_url': base_url + tds[9].a['href'] if tds[9].a else '',
            'classroom_url': tds[11].a['href'] if tds[11].a else '',
            'classroom': tds[11].a.get_text().replace('(', '').replace(')', '') if tds[11].a else '',
        })

    return row_dicts


def create_dept(filename):
    with open(filename) as f:
        soup = BeautifulSoup(f.read().replace(u'\xa0', u''), 'html.parser')
        rows = generate_rows(soup)
        for row in rows:
            try:
                Course.objects.update_or_create(
                    semester=filename.split('/')[1],
                    serial_no=row['serial_no'],
                    defaults=row
                )
            except Exception as e:
                print ('%s-%s - %s' % (row['serial_no'], row['title'], e))


if __name__ == '__main__':
    list_dirs = os.walk('crawler/html')
    filenames = []
    for root, dirs, files in list_dirs:
        for f in files:
            if '.html' in f:
                filenames.append(os.path.join(root, f))
    print (filenames)
    for filename in filenames:
        create_dept(filename)
