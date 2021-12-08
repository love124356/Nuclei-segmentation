import json

json_format = {}
annotations_path = './dataset/coco/annotations'
with open(annotations_path + "/test_img_ids.json", "r") as f:
    image = json.load(f)

count = 0
json_format['images'] = image
json_format['categories'] = [
    {"supercategory": "nuclei", "name": "nuclei", "id": 1}]

json_object = json.dumps(json_format)
with open(annotations_path + "/test.json", "w") as outfile:
    outfile.write(json_object)

print("DONE.")
