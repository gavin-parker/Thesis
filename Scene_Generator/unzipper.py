import json
import os
table_objects = ["bowl", "bottle", "can", "computer keyboard", "keypad", "display", "phone", "jar", "knife", "lamp", "laptop", "microphone"
                 "mug", "remote", "wine bottle"]
data = json.load(open('taxonomy.json'))

for datapoint in data:
    for obj in table_objects:
        if obj in datapoint['name']:
            os.system('unzip {}.zip'.format(datapoint['synsetId']))
            break