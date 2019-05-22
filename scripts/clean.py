import json
inp = open("data.json", "rb")
passages = json.load(inp)
inp.close()


for passage in passages:
    passage["passage"] = passage["passage"].replace("九个亿财经消息——", "")
    passage["passage"] = passage["passage"].replace("本文为九个亿财经翻译稿件，未经授权不得转载", "")
    passage["passage"] = passage["passage"].replace("稿件来源：九个亿财经", "")
    passage["passage"] = passage["passage"].replace("已经协议授权的媒体下载使用时须注明", "")
    passage["passage"] = passage["passage"].replace("违者将依法追究责任", "")

outp = open("data.json", 'w', encoding="utf-8")
outp.write(json.dumps(passages, indent=4, ensure_ascii=False))
outp.close()