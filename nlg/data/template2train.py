import json

ratio = {'train':5, 'valid':1, 'test':1}

train = []
test = []
valid = []

# raw_data = json.load(open('dialogue_log.json','r'))
"""
# Label data
for i in range(len(raw_data)):
    dact = raw_data[i]['intent']
    slot = '('
    for j in range(len(raw_data[i]['labels'])):
        if raw_data[i]['labels'][j] != 'O':
            if j > 1 and raw_data[i]['labels'][j] == raw_data[i]['labels'][j-1]:
                slot = slot[:-2] + raw_data[i]['tokens'][j] + "';"
            else:
                slot = slot + raw_data[i]['labels'][j][2:] + "='"
                slot = slot + raw_data[i]['tokens'][j] + "';"

    if slot[-1] == ';':
        slot = slot[:-1] + ')'
    else:
        slot = slot + ')'

    dact = dact + slot
    hres = ' '.join(raw_data[i]['tokens'])

    div = ratio['train'] + ratio['test'] + ratio['valid']
    if i % div < ratio['train']:
        train.append([dact,hres,hres])
    elif i % div < ratio['train'] + ratio['test']:
        test.append([dact,hres,hres])
    else:
        valid.append([dact,hres,hres])
"""

# Template Data
raw_data = json.load(open('template.json','r'))
for i in range(len(raw_data)):
    dact = raw_data[i]['intent']
    slot = '('
    for j in range(min(len(raw_data[i]['labels']), len(raw_data[i]['tokens']))):
        if raw_data[i]['labels'][j] != 'O' and raw_data[i]['labels'][j] != 'o':
            if j > 1 and raw_data[i]['labels'][j] == raw_data[i]['labels'][j-1]:
                slot = slot[:-2] + raw_data[i]['tokens'][j] + "';"
            else:
                slot = slot + raw_data[i]['labels'][j][2:] + "='"
                slot = slot + raw_data[i]['tokens'][j] + "';"

    if slot[-1] == ';':
        slot = slot[:-1] + ')'
    else:
        slot = slot + ')'

    dact = dact + slot
    hres = ' '.join(raw_data[i]['tokens'])

    div = ratio['train'] + ratio['test'] + ratio['valid']
    train.append([dact,hres,hres])

trainfile = open('train.json','w')
#validfile = open('valid.json','w')
#testfile = open('test.json','w')

json.dump(train, trainfile, ensure_ascii=False, indent=4)
#json.dump(test, testfile, ensure_ascii=False, indent=4)
#json.dump(valid, validfile, ensure_ascii=False, indent=4)

