import concurrent.futures
import os
import time
import requests
import urllib
from bs4 import BeautifulSoup
from const import ntu_course_dpt_search_url


def crawl_course(dept_id, select_sem):
    post_data = ('current_sem=%s&'
                 'dpt_sel=%s&'
                 'dptname=%s&'
                 'yearcode=0&'
                 'selcode=-1&'
                 'coursename=&'
                 'teachername=&'
                 'alltime=yes&'
                 'allproced=yes&'
                 'allsel=yes&'
                 'page_cnt=1500' % (select_sem, dept_id[0] + '000', dept_id))
    data = dict(urllib.parse.parse_qs(post_data))
    r = requests.post(ntu_course_dpt_search_url, data=data)

    trs_str = ''
    if r.status_code == 200:
        soup = BeautifulSoup(r.text.encode('latin1', 'ignore').decode('big5', 'ignore'), 'html.parser')
        trs = soup.find(bordercolorlight="#CCCCCC").find_all('tr')[1:]
        trs_str = ''.join(map(str, trs))
    return r.status_code, trs_str, dept_id, select_sem


def crawl_college(ntu_soup):
    start = time.time()
    ntu_colls = {}
    coll_tags = ntu_soup.find('select', id='dpt_sel')
    for coll_tag in coll_tags.find_all('option'):
        coll_str = coll_tag.string.strip().split(' ')
        coll_name = coll_str[1] if len(coll_str) > 1 else coll_str[0]
        ntu_colls[coll_tag['value']] = coll_name
    end = time.time()
    print('Elapsed Time: ' + str(end - start) + '\n')
    # print(ntu_colls)
    return ntu_colls


def crawl_depts(ntu_soup):
    start = time.time()
    ntu_depts = {}
    dept_tags = ntu_soup.find_all(id='dptname')[0]
    for dept_tag in dept_tags.find_all('option'):
        dept_str = dept_tag.string.strip().split(' ')
        dept_name = dept_str[1] if len(dept_str) > 1 else dept_str[0]
        ntu_depts[dept_tag['value']] = dept_name
    end = time.time()
    print('Elapsed Time: ' + str(end - start) + '\n')
    # print(ntu_depts)
    return ntu_depts


def crawl_sems(ntu_soup):
    start = time.time()
    ntu_sems = []
    sem_tags = ntu_soup.find_all(id='select_sem')[0]
    for sem_tag in sem_tags.find_all('option'):
        ntu_sems.append(sem_tag['value'])
    end = time.time()
    print('Elapsed Time: ' + str(end - start) + '\n')
    # print(ntu_sems)
    return ntu_sems


def create_dirs(ntu_depts, ntu_sems):
    if not os.path.exists('./html'):
        os.makedirs('./html')
    for dept_id, dept_name in ntu_depts.items():
        for sem in ntu_sems:
            if not os.path.exists(('./html/%s' % sem)):
                os.makedirs(('./html/%s' % sem))


def check_empty_html(ntu_depts, ntu_sems):
    count = 0
    if not os.path.exists('./html'):
        os.makedirs('./html')
    for dept_id, dept_name in ntu_depts.items():
        for sem in ntu_sems:
            empty = False
            if not os.path.exists(('./html/%s' % sem)):
                os.makedirs(('./html/%s' % sem))
            try:
                with open(('./html/%s/%s.html' % (sem, dept_id)), 'r') as f:
                    content = f.read()
                    if len(content) == 0:
                        empty = True
                if empty:
                    count += 1
                    print(('./html/%s/%s.html is empty!!!' % (sem, dept_id)))
            except:
                print(('./html/%s/%s.html doesn\'t exist.' % (sem, dept_id)))
    print('Total %d html files are empty!' % count)


def main():
    ntu_res = requests.get(ntu_course_dpt_search_url, stream=True)
    ntu_soup = BeautifulSoup(ntu_res.text.encode(ntu_res.encoding, 'ignore').decode('big5', 'ignore'), 'html.parser')

    ntu_colls = crawl_college(ntu_soup)
    ntu_depts = crawl_depts(ntu_soup)
    ntu_sems = crawl_sems(ntu_soup)
    create_dirs(ntu_depts, ntu_sems)
    check_empty_html(ntu_depts, ntu_sems)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ntu_sems)) as executor:
        future_to_crawl = {}
        for dept_id, dept_name in ntu_depts.items():
            for sem in ntu_sems:
                reply_str = '%s %s %s ' % (sem, dept_id, dept_name)
                if not os.path.exists(('./html/%s/%s.html' % (sem, dept_id))):
                    future_to_crawl[executor.submit(crawl_course, dept_id, sem)] = reply_str

        for future in concurrent.futures.as_completed(future_to_crawl):
            reply_str = future_to_crawl[future]
            try:
                status_code, trs_str, dept_id, sem = future.result()
            except Exception as exc:
                print('%s crawled an exception: %s' % (reply_str, exc))
            else:
                if status_code == 200:
                    with open(('./html/%s/%s.html' % (sem, dept_id)), 'w') as f:
                        f.write(trs_str)
                    print(reply_str + 'crawled.')
                else:
                    print(reply_str + 'failed.')
                    failed.append(reply_str)
    print(failed)


if __name__ == '__main__':
    main()
