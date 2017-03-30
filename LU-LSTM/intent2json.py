# coding: utf-8
import json

def UserIntentJson(UserIntent):
    output = {}
    slot = {}
    slot_name = set(UserIntent[2]).difference('O')
    output['intent'] = UserIntent[0]
    for target in slot_name:
        slot[target] = UserIntent[1][UserIntent[2].index(target)]
    output['slot'] = slot
    return json.dumps(output)
    




if __name__ == '__main__':
    import codecs
    with codecs.open('test','r','utf-8') as f:
        for i in range(15):
            test_input = (f.readline().split()[0],f.readline().split(),f.readline().split())
            result = UserIntentJson(test_input)
            print(json.loads(result))

