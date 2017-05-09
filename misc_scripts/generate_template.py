# coding: utf-8
from access_django import *
import random
from django.template import Context, Template
from utils.nlg import *
from utils.tagger import *
from utils.query import *
# Templates
# possible_slots = [
# 'title',
# 'when',
# 'instructor',
# 'classroom',
# 'designated_for',
# 'required_elective',
# 'sel_method'
# ]
be = ['', '是']
ask = ['', '請問', '告訴我', '跟我說', '想知道']

templates = {
    'request_title': [
        Template('{{ask}}有開哪些課'),
        Template('{{ask}}有哪些課'),
        Template('{{ask}}有開什麼課'),
        Template('{{ask}}{{instructor}}在{{when}}有哪些課'),
        Template('{{ask}}{{instructor}}有開哪些課'),
        Template('{{ask}}{{instructor}}有開什麼課'),
        Template('{{ask}}{{designated_for}}在{when}}有哪些課'),
        Template('{{ask}}{{designated_for}}有開哪些課'),
        Template('{{ask}}{{designated_for}}{{instructor}}老師有開哪些課'),
        Template('{{ask}}{{when}}有哪些課'),
        Template('{{ask}}{{when}}{{designated_for}}有開哪些課'),
        Template('{{ask}}加簽方式是{{sel_method}}的課有哪些'),
        Template('{{ask}}{{when}}加簽方式是{{sel_method}}的課有哪些'),
        Template('{{ask}}{{when}}在{{classroom}}上課的是哪一門課'),
    ],
    'request_instructor': [
        Template('{{ask}}老師是誰'),
        Template('{{ask}}老師是哪位'),
        Template('{{ask}}開{{title}}的老師是哪位'),
        Template('{{ask}}是誰開的'),
        Template('{{ask}}{{title}}這堂課的老師是哪位'),
        Template('{{ask}}哪位教授開的'),
        Template('{{ask}}教授是誰'),
        Template('{{ask}}是誰教的'),
        Template('{{ask}}{{designated_for}}的{{title}}是誰上的'),
        Template('{{ask}}{{designated_for}}的{{title}}是誰教的'),
        Template('{{ask}}{{designated_for}}的{{title}}是哪位教授'),
        Template('{{ask}}{{when}}{{designated_for}}的{{title}}是誰上的'),
        Template('{{ask}}{{when}}{{designated_for}}的{{title}}是誰教的'),
        Template('{{ask}}{{when}}{{designated_for}}的{{title}}是哪位教授'),

    ],
    'request_schedule_str': [
        Template('什麼時候的課'),
        Template('上課時間在什麼時候'),
        Template('什麼時候上課'),
        Template('星期幾上課'),
        Template('幾點上課'),
        Template('第幾節的課'),
        Template('上課時間'),
        Template('在第幾節'),
        Template('{{title}}的上課時間'),
        Template('{{title}}幾點上課'),
        Template('{{title}}在哪天上課'),
        Template('{{title}}是星期幾的課'),
        Template('{{title}}是第幾節的課'),
        Template('{{title}}課上課時間'),
        Template('{{title}}課幾點上課'),
        Template('{{title}}課在哪天上課'),
        Template('{{title}}課是星期幾的課'),
        Template('{{title}}課是第幾節的課'),
        Template('星期幾有{{title}}'),
        Template('什麼時候有{{title}}'),
        Template('星期幾有{{title}}課'),
        Template('什麼時候有{{title}}課'),
        Template('星期幾有{{title}}的課'),
        Template('{{instructor}}在星期幾有課'),
        Template('星期幾有{{instructor}}老師的課'),
        Template('什麼時候有{{instructor}}老師的課'),
        Template('{{instructor}}老師的{{title}}課在什麼時候'),
        Template('{{instructor}}老師的{{title}}課在什麼時候上課'),
        Template('{{instructor}}老師的{{title}}課在星期幾'),
        Template('{{instructor}}老師的{{title}}課在幾點'),
        Template('{{instructor}}老師的{{title}}課在什麼時間'),
        Template('{{instructor}}老師的{{title}}課在什麼時間上課'),
    ],
    'request_classroom': [
        Template('在哪裡上課'),
        Template('教室在哪'),
        Template('哪間教室'),
        Template('在哪邊上'),
        Template('{{title}}在哪裡上課'),
        Template('{{title}}課在哪裡上課'),
        Template('{{title}}的上課教室在哪'),
        Template('{{title}}課的上課教室在哪'),
        Template('{{title}}的教室在哪'),
        Template('{{title}}課的教室在哪'),
        Template('{{title}}課的教室'),
        Template('{{title}}的教室'),
        Template('{{title}}的上課地點'),
        Template('{{title}}的地點'),
        Template('在那裡上{{title}}'),
        Template('{{instructor}}的課在哪間教室'),
        Template('{{instructor}}的課在哪裡'),
        Template('{{instructor}}老師{{when}}的課在哪間教室'),
        Template('{{instructor}}老師{{when}}的課在哪裡'),
        Template('{{instructor}}老師{{when}}的課在哪間教室上課'),
        Template('{{instructor}}老師{{when}}的課在哪裡上課'),
        Template('{{instructor}}教授{{when}}的課在哪間教室'),
        Template('{{instructor}}教授{{when}}的課在哪裡'),
        Template('{{instructor}}教授{{when}}的課在哪間教室上課'),
        Template('{{instructor}}教授{{when}}的課在哪裡上課'),
        Template('{{when}}{{instructor}}老師的課在哪間教室'),
        Template('{{when}}{{instructor}}老師的課在哪裡上課'),
        Template('{{when}}{{instructor}}老師的課在哪裡'),
        Template('{{when}}{{instructor}}教授的課在哪間教室'),
        Template('{{when}}{{instructor}}教授的課在哪裡上課'),
        Template('{{when}}{{instructor}}教授的課在哪裡')

    ],
    'request_review': [
        Template('{{title}}的評價如何'),
        Template('有沒有{{title}}的評價'),
        Template('{{instructor}}的評價如何'),
        Template('這堂課怎麼樣'),
        Template('{{title}}這堂課的評價怎樣'),
        Template('{{title}}的評價好嗎'),
        Template('大家怎麼看{{title}}這堂課呢'),
        Template('{{title}}這堂課好不好'),
        Template('大家修{{title}}這堂課的評價如何'),
        Template('有討論{{title}}的文章嗎'),
        Template('有討論{{title}}這堂課的文章嗎'),
        Template('{{instructor}}老師的評價如何'),
        Template('大家覺得{{instructor}}老師的評價如何'),
        Template('大家覺得{{instructor}}老師的評價怎樣'),
        Template('{{instructor}}老師的評價'),
        Template('有討論{{instructor}}老師文章嗎')
    ],
    'request_designated_for': [
        Template('{{title}}是哪個系開的'),
        Template('{{title}}是哪個系所開的'),
        Template('什麼系開的'),
        Template('什麼系所的課'),
        Template('{{title}}是哪個系的課'),
        Template('{{title}}是哪個系所的課'),
        Template('哪個系開{{title}}這堂課'),
        Template('哪個系開{{title}}'),
        Template('哪個系所開{{title}}這堂課'),
        Template('哪個系所有開{{title}}這堂課'),
        Template('{{instructor}}在哪個系有開課'),
        Template('{{instructor}}在哪個系所有開課')

    ],
    'request_required_elective': [
        Template('是必修還選修'),
        Template('是{{designated_for}}必修嗎?'),
        Template('{{designated_for}}的必修嗎?'),
        Template('{{title}}是{{designated_for}}的必修嗎'),
        Template('{{title}}是{{designated_for}}的必修課嗎'),
        Template('{{title}}是必修嗎'),
        Template('{{instructor}}有開必修課嗎'),
        Template('{{when}}有{{designated_for}}的必修課嗎'),
        Template('有{{instructor}}開的必修課嗎')

    ],
    'request_sel_method': [
        Template('{{title}}的加選方式是什麼'),
        Template('{{title}}的加選方式?'),
        Template('{{title}}如何加選?'),
        Template('{{title}}要怎麼加選?'),
        Template('{{title}}怎樣加簽?'),
        Template('{{instructor}}會開放加簽嗎?'),
        Template('{{instructor}}有開放加簽嗎'),
        Template('{{instructor}}會不會簽人'),
        Template('{{instructor}}通常會不會加簽'),
        Template('{{title}}這堂課要怎麼加簽?'),
        Template('{{title}}這堂課的加選方式是什麼'),
        Template('{{title}}這堂課的加選方式?'),
        Template('{{title}}這堂課如何加選?'),
        Template('{{title}}這堂課要怎麼加選?'),
        Template('{{title}}這堂課怎樣加簽?'),
        Template('{{title}}怎樣加簽?')
    ],
    'inform': [
        Template('{{be}}{{title}}'),
        Template('叫{{title}}'),
        Template('叫做{{title}}'),
        Template('{{title}}喔'),
        Template('{{be}}{{when}}'),
        Template('{{be}}{{when}}的課'),
        Template('{{be}}{{when}}上課'),
        Template('{{be}}{{when}}上的'),
        Template('{{be}}{{instructor}}'),
        Template('{{be}}{{instructor}}老師'),
        Template('{{be}}{{instructor}}教授'),
        Template('{{be}}{{instructor}}的課'),
        Template('{{be}}{{classroom}}'),
        Template('在{{classroom}}上課'),
        Template('上課地點是{{classroom}}'),
        Template('教室在{{classroom}}'),
        Template('{{be}}{{designated_for}}'),
        Template('{{be}}{{designated_for}}開的'),
        Template('{{be}}{{designated_for}}的課'),
        Template('{{be}}{{designated_for}}上的'),
        Template('{{be}}{{required_elective}}'),
        Template('{{be}}{{required_elective}}課'),
        Template('{{be}}{{required_elective}}的課'),
        Template('{{be}}{{sel_method}}'),
        Template('加選方法{{be}}{{sel_method}}'),
        Template('選課方法{{be}}{{sel_method}}'),
        Template('{{be}}{{sel_method}}'),
    ],
    'inform_unknown': [   # prevent infinite loop
        Template('不知道'),
        Template('我不知道'),
        Template('不知道耶'),
        Template('我真的不知道啦'),
        Template('不清楚'),
        Template('我不清楚'),
        Template('我真的不清楚'),
        Template('不知'),
        Template('不確定'),
        Template('我知道還要問你嗎'),
        Template('不要問我'),
        Template('別問我'),
        Template('我怎麼知道')
    ],
    'other': [
        Template('快被當了怎模辦QQ'),
        Template('什麼時候可以停修QQ'),
        Template('學海無涯，回頭是岸'),
        Template('想畢業'),
        Template('我不想上課QQ'),
        Template('想耍廢'),
        Template('我好廢怎麼辦'),
        Template('人生好困難'),
        Template('智商不夠好痛苦'),
        Template('通識搶不到怎麼辦'),
        Template('你可以聰明一點嗎'),
        Template('什麼時候期末考'),
        Template('羨慕大神人生都沒有挫折'),
        Template('好魯'),
        Template('呵呵'),
        Template('我啥都不會'),
        Template('智商不夠可以砍掉重來嗎QQ'),
        Template('ㄎㄅ'),
        Template('是個擅長講幹話的朋友呢'),
        Template('不想看紙'),
        Template('沒錢'),
        Template('給我一把人生重來槍'),
        Template('午餐吃啥'),
        Template('晚餐吃啥'),
        Template('早餐吃啥'),
        Template('冰咖啡買一送一不加糖'),
        Template('這真的可以嗎'),
        Template('祐婷大大凱瑞眾生'),
        Template('謝謝大大分享'),
        Template('我好邊緣'),
        Template('樓主一生平安順利'),
        Template('一號餐要薯餅冰那提少冰加糖'),
        Template('麥當當好吃'),
    ]
}


def trim_course(course):
    course['when'] = random.choice(['星期', '禮拜']) + course['schedule_str'][0]
    course['be'] = random.choice(be)
    course['ask'] = random.choice(ask)
    for k in ['title', 'instructor', 'classroom']:
        course[k] = trim_attr(course[k])
    return course

if __name__ == '__main__':
    """
    In this format:
    Line1: Intent
    Line2: Tokenized sentence
    Line3: BIO
    ======
    classroom
    禮拜三 高分子材料概論 在 哪個 系館 哪間 教室 上課 嗎 ?
    B_when B_title O O O O O O O O
    title
    幫 我 找 吳俊傑 教 哪些 課 ?
    O O O B_instructor O O O O
    """
    print('[Info] Start generating templates')
    # TODO Change to argparse
    filename = 'training_template.txt'
    N = 1000
    courses = query_course({}).values()  # Get all course
    # TODO Refine request_schedule_str to when
    #
    with open(filename, 'w') as f:
        for intent, tpls in templates.items():
            for tpl in tpls:
                for _ in range(N):
                    course = random.choice(courses)
                    course = trim_course(course)
                    # Jieba cut sentence
                    sentence = ' '.join(cut(tpl.render(Context(course))))
                    # BIO tagged sentence
                    bio_tagged = ' '.join(BIO(sentence, course))

                    f.write(intent + '\n')
                    f.write(sentence + '\n')
                    f.write(bio_tagged + '\n')
