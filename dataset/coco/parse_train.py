import json
import shutil
import numpy as np
import cv2
import os
from pycocotools import mask
from skimage import measure


def setAnno(img_path, idx, count):

    ground_truth_binary_mask = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    fortran_ground_truth_binary_mask = np.asfortranarray(
                                        ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": idx,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": count
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    return annotation


json_format = {"annotations": [], 'images': []}
train_path = './dataset/coco/train'
trainval_path = './dataset/coco/trainval'
annotations_path = './dataset/coco/annotations'
trainval_dir = os.makedirs(trainval_path, exist_ok=True)
anns_dir = os.makedirs(annotations_path, exist_ok=True)
imgs_folder = os.listdir(train_path)
masks = []
count = 0
for _folder in imgs_folder:
    idx = imgs_folder.index(_folder)
    print("Parse training image ", idx + 1)
    full_path = os.path.join(train_path, _folder)
    mask_path = os.path.join(full_path, "masks")
    imgs_path = os.path.join(full_path, "images")

    # move img to trainval folder
    img_name = os.listdir(imgs_path)[0]
    img_path = os.path.join(imgs_path, img_name)
    new_img_path = os.path.join(trainval_path, img_name)
    shutil.copyfile(img_path, new_img_path)

    # set images
    height, width = cv2.imread(new_img_path).shape[:2]
    image_format = {
            "file_name": img_name,
            "id": idx + 1,
            "height": height,
            "width": width,
        }
    json_format['images'].append(image_format)
    # mask to coco
    masks_img = [_ for _ in os.listdir(mask_path) if _.endswith(".png")]
    for img in masks_img:
        count = count + 1
        path = os.path.join(mask_path, img)
        json_format["annotations"].append(setAnno(path, idx+1, count))

    # count = count + len(masks_img)

json_format['categories'] = [
    {"supercategory": "nuclei", "name": "nuclei", "id": 1}]

json_object = json.dumps(json_format)
with open(annotations_path + "/train.json", "w") as outfile:
    outfile.write(json_object)
print("DONE.")
