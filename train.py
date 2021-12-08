import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

setup_logger()
register_coco_instances(
    "nuclei_dataset_train", {},
    "dataset/coco/annotations/train.json", "dataset/coco/trainval")

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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MIN_SIZE_TRAIN = 1200
cfg.INPUT.MAX_SIZE_TRAIN = 1500
cfg.INPUT.MIN_SIZE_TEST = 1200
cfg.INPUT.MAX_SIZE_TEST = 1500

count = 1
makedir = False
while(not makedir):
    output_path = './output/R101/' + str(count)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        makedir = True
    else:
        count = count + 1

cfg.OUTPUT_DIR = output_path

print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Save model.pt in " + output_path)
