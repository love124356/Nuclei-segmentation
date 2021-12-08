import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os
import json
import cv2
from tqdm import tqdm
import pycocotools.mask as mask_util

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

setup_logger()

register_coco_instances(
    "nuclei_dataset_train", {},
    "dataset/coco/annotations/train.json", "dataset/coco/trainval")
register_coco_instances(
    "nuclei_dataset_test", {},
    "dataset/coco/annotations/test.json", "dataset/coco/test")

dataset_metadata = MetadataCatalog.get("nuclei_dataset_test")
dataset_dicts = DatasetCatalog.get("nuclei_dataset_test")

# Inference should use the config with parameters that are used in training
cfg = get_cfg()
cfg.merge_from_file(
    "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("nuclei_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class.
cfg.INPUT.MIN_SIZE_TRAIN = 1200
cfg.INPUT.MAX_SIZE_TRAIN = 1500
cfg.INPUT.MIN_SIZE_TEST = 1200
cfg.INPUT.MAX_SIZE_TEST = 1500

# NUM = 1
# output_path = './output/R101/' + str(NUM)
# cfg.OUTPUT_DIR = output_path

# We changed it a little bit for inference
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join("model/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.TEST.DETECTIONS_PER_IMAGE = 600

# print(cfg)
predictor = DefaultPredictor(cfg)

total_instances = []
prediction = []
for img in tqdm(dataset_dicts):
    id = img["image_id"]
    im = cv2.imread(img["file_name"])

    outputs = predictor(im)
    # print(outputs)
    instances = outputs["instances"].to("cpu")
    instance_num = len(instances)
    total_instances.append(instance_num)
    categories = instances.pred_classes
    masks = instances.pred_masks
    scores = instances.scores

    # RLE
    instances.pred_masks_rle = [
        mask_util.encode(np.asfortranarray(mask)) for mask in masks]
    for rle in instances.pred_masks_rle:
        rle['counts'] = rle['counts'].decode('utf-8')

    instances.remove('pred_masks')

    for i in range(instance_num):
        pred = {}
        pred['image_id'] = id
        bboxs = instances.pred_boxes[i].tensor.numpy().tolist()[0]
        left = bboxs[0]
        top = bboxs[1]
        w = bboxs[2] - bboxs[0]
        h = bboxs[3] - bboxs[1]
        pred['bbox'] = [left, top, w, h]
        pred['score'] = float(scores[i])
        pred['category_id'] = int(categories[i]) + 1
        pred['segmentation'] = instances.pred_masks_rle[i]
        prediction.append(pred)

    # TO TEST INVERT CONVERSION
    # instances.pred_masks = np.stack(
    #     [mask_util.decode(rle) for rle in instances.pred_masks_rle])
    # print(instances.pred_masks)
json_object = json.dumps(prediction, indent=4)
with open("answer.json", "w") as outfile:
    outfile.write(json_object)

print(total_instances)
print("Total instances: ", sum(total_instances))
print("DONE. answer.json is saved in root.")
