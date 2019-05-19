import json
import random
 
def openJson():
    inp = open("data.json", "rb")
    passages = json.load(inp)
    inp.close()
    return passages

def read():
    t = input()
    if t != "0" and t !="1" and t != "2":
        t = input()
    return t

neg_count = 0
pos_count = 0
passages = openJson()
print(len(passages))
for passage in passages:
    if 'label' in passage.keys():
        if passage['label'] == 0:
            neg_count += 1
        else:
            pos_count += 1
print(pos_count, " ", neg_count)
count = 0

while True:
    index = random.randint(0, len(passages) - 1)
    if 'label' in passages[index].keys():
        continue
    print("No.%d" % (count + 1))
    print(passages[index]["title"])
    print(passages[index]["passage"])
    print("1:Positive 0:Negative 2:skip")
    i = int(read())
    if i == 2:
        continue
    sen = i
    passages[index]["label"] = sen
    count += 1
    if count == 100:
        outp = open("data.json", 'w', encoding="utf-8")
        outp.write(json.dumps(passages, indent=4, ensure_ascii=False))
        outp.close()
        print("1(2):Continue 0:End")
        if int(read()) != 0:
            passages = openJson()
            count = 0
        else:
            break