from crawler.models import *
import jieba


def dictionary_create(save_name) :
    lst_instructor = []
    lst_title = []
    lst_1 = ['禮拜一','禮拜二','禮拜三','禮拜四','禮拜五','禮拜六','禮拜日','禮拜天']
    Course_all = Course.objects.all()

    for course in Course_all :
        if course.instructor not in lst_instructor :
            lst_instructor.append(course.instructor)
        if course.title not in lst_title :
            lst_title.append(course.title)

    with open(save_name , 'w') as f:
        for item in lst_instructor :
            f.write('%s 99999\n' % item)
        for item in lst_title :
            f.write('%s 99999\n' % item)
        for item in lst_1 :
            f.write('%s 99999\n' % item)

def seg(dic_name,sentence) :
    jieba.load_userdict(dic_name)
    seg_list = jieba.cut(sentence, cut_all=False)
    output = " ".join(seg_list)
    return output


if __name__ == '__main__' :
    #dictionary_create('dictionary.txt')

    # for test
    sentence = '幫我查禮拜天陳信希老師的課'
    output = seg('dictionary.txt',sentence)
    print(output)
