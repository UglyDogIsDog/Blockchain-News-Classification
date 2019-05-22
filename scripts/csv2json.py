import csv
import json
import sys

csv_file = open("test_data.csv", "r", encoding="utf-8")
json_file = open("test.json", 'w', encoding="utf-8")

field_names = ("id", "date", "company", "code", "label", "title", "passage")
reader = csv.DictReader(csv_file, field_names)
l = []
for i, row in enumerate(reader):
    if i > 0: #remove first line
        row["label"] = int(row["label"])
        l.append(row)
    
json_file.write(json.dumps(l, indent=4, ensure_ascii=False))

csv_file.close()
json_file.close()