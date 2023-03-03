import os
from typing_extensions import Literal
from typing import List, Any, Dict
import cv2
from dotenv import load_dotenv
import torch
import supervisely as sly

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

weights_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# code for detectron2 inference copied from official Colab tutorial (inference section):
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html


class MyModel(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        # Here we are downloading model weights by url for demo,
        # but it also can be downloaded from Team Files (passing an entire folder is possible too)
        # weights_path = self.download(weights_url)
        weights_path = "my_model/weights.pkl"
        model_info = sly.json.load_json_file(os.path.join(self.model_dir, "model_info.json"))
        architecture_name = model_info["architecture"]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(architecture_name))
        cfg.MODEL.DEVICE = device  # learn more in torch.device
        cfg.MODEL.WEIGHTS = weights_path
        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionMask]:
        confidence_threshold = settings.get("confidence_threshold", 0.5)
        image = cv2.imread(image_path)  # BGR

        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        outputs = self.predictor(image)  # get predictions from Detectron2 model
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            # filter predictions by confidence
            if score >= confidence_threshold:
                results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results


model_dir = "my_model"  # model weights will be downloaded into this dir
settings = {"confidence_threshold": 0.7}
m = MyModel(model_dir=model_dir, custom_inference_settings=settings)
m.load_on_device(model_dir=model_dir, device=device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
