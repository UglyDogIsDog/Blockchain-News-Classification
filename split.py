import json
import random

inp = open("data.json", "rb")
passages = json.load(inp)
inp.close()

l = []
for passage in passages:
    if 'label' in passage.keys():
        l.append(passage)

random.shuffle(l)

print(len(l))

outp = open("train.json", 'w', encoding="utf-8")
outp.write(json.dumps(l[ : len(l) - 100], indent=4, ensure_ascii=False))
outp.close()

outp = open("test.json", 'w', encoding="utf-8")
outp.write(json.dumps(l[len(l) - 100 : ], indent=4, ensure_ascii=False))
outp.close()