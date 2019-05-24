import json
import torch

inp = open("data.json", "rb")
passages = json.load(inp)
print(len(passages))
inp.close()

label = torch.load("test.json.lab", map_location='cpu')
print(len(label))

i = 0
for passage in passages:
    passage["label"] = label[i].item()
    i += 1

outp = open("final.json", 'w', encoding="utf-8")
outp.write(json.dumps(passages, indent=4, ensure_ascii=False))
outp.close()

print(label[0].item())
print(label[1].item())
print(label[2].item())
print(label[3].item())
print(label[4].item())