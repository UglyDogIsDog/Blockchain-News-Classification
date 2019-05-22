import json
import random
 
def openJson():
    inp = open("data.json", "rb")
    passages = json.load(inp)
    inp.close()
    return passages

count = 0
passages = openJson()
while True:
    index = random.randint(0, len(passages))
    if 'label' in passages[index].keys():
        continue
    print(passages[index]["title"])
    print(passages[index]["passage"])
    print("1:Positive 0:Negative")
    sen = int(input())
    passages[index]["label"] = sen
    count += 1
    if count == 20:
        outp = open("data.json", 'w', encoding="utf-8")
        outp.write(json.dumps(passages, indent=4, ensure_ascii=False))
        outp.close()
        print("1:Continue 0:End")
        if int(input()) == 1:
            passages = openJson()
            count = 0
        else:
            break