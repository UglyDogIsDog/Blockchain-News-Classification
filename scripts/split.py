import json
import random

inp = open("data.json", "rb")
passages = json.load(inp)
inp.close()

l = []
neg_count = 0
pos_count = 0
for passage in passages:
    if 'label' in passage.keys():
        if passage['label'] == 0:
            neg_count += 1
        if passage['label'] == 1:
            pos_count += 1
        p = {}
        p['passage'] = passage['passage']
        p['label'] = passage['label']
        l.append(p)

random.shuffle(l)

print("%d %d %d" % (len(l), neg_count, pos_count))

outp = open("train.json", 'w', encoding="utf-8")
outp.write(json.dumps(l[ : int(len(l) / 5 * 4)], indent=4, ensure_ascii=False))
outp.close()

outp = open("test.json", 'w', encoding="utf-8")
outp.write(json.dumps(l[int(len(l) / 5 * 4) : ], indent=4, ensure_ascii=False))
outp.close()
