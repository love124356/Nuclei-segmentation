import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import os
import cv2

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

setup_logger()

register_coco_instances(
    "nuclei_dataset_train", {},
    "dataset/coco/annotations/train.json", "dataset/coco/trainval")
dataset_metadata = MetadataCatalog.get("nuclei_dataset_train")
dataset_dicts = DatasetCatalog.get("nuclei_dataset_train")
print(dataset_metadata)
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1],
                            metadata=dataset_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    name = d["file_name"].split("/")[-1]
    dir = os.path.join('input_img', name)
    os.makedirs('input_img/', exist_ok=True)
    cv2.imwrite(dir, out.get_image()[:, :, ::-1])
