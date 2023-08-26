import json

with open('train.json', 'r') as file:
    data = json.load(file)

    if isinstance(data, list):
        for item in data[:4]:
            print(item)
