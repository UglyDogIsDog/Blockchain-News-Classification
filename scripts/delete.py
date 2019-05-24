import json


inp = open("data.json", "rb")
passages = json.load(inp)

l = []
for passage in passages:
    p = {}
    p['title'] = passage['title']
    p['date'] = passage['date']
    p['passage'] = passage['passage']
    p['label'] = 0
    l.append(p)


inp.close()
outp = open("test.json", 'w', encoding="utf-8")
outp.write(json.dumps(l, indent=4, ensure_ascii=False))
outp.close()