import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import os
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
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

NUM = 1
output_path = './output/R101/' + str(NUM)
cfg.OUTPUT_DIR = output_path

# We changed it a little bit for inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.TEST.DETECTIONS_PER_IMAGE = 600

# print(cfg)
predictor = DefaultPredictor(cfg)

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    name = d["file_name"].split("/")[-1]
    dir = os.path.join('output_img/R101/', str(NUM), 'pred_0.05_' + name)
    os.makedirs('output_img/R101/' + str(NUM), exist_ok=True)
    cv2.imwrite(dir, out.get_image()[:, :, ::-1])
