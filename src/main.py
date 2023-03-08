import os
from pathlib import Path
from typing_extensions import Literal
from typing import List, Any, Dict
import cv2
from dotenv import load_dotenv
import supervisely as sly
import supervisely.nn.inference.gui as GUI

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

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
        self.gui: GUI.InferenceGUI
        selected_model = self.gui.get_checkpoint_info()
        weights_path, config_path = self.download_pretrained_files(selected_model, model_dir)

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = device  # learn more in torch.device
        cfg.MODEL.WEIGHTS = weights_path
        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def support_custom_models(self) -> bool:
        return False

    def get_models(self) -> List[Dict[str, str]]:
        mask_rcnn_R_50_C4_1x = {
            "Model": "R50-C4 (1x)",
            "Dataset": "COCO",
            "Train_time": "0.584",
            "Inference_time": "0.110",
            "Box AP score": "36.8",
            "Mask AP score": "32.2",
        }

        mask_rcnn_R_50_DC5_3x = {
            "Model": "R50-DC5 (3x)",
            "Dataset": "COCO",
            "Train_time": "0.470",
            "Inference_time": "0.076",
            "Box AP score": "40.0",
            "Mask AP score": "35.9",
        }
        return [mask_rcnn_R_50_C4_1x, mask_rcnn_R_50_DC5_3x]

    def get_models_url(self):
        return {
            "R50-C4 (1x)": {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl",
            },
            "R50-DC5 (3x)": {
                "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
                "weightsUrl": "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl",
            },
        }

    def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
        models_url = self.get_models_url()
        weights_url = models_url[selected_model["Model"]]["weightsUrl"]
        config_path = os.path.join(
            model_dir, "configs", models_url[selected_model["Model"]]["config"]
        )
        weights_ext = sly.fs.get_file_ext(weights_url)
        weights_dst_path = os.path.join(model_dir, f"{selected_model['Model']}{weights_ext}")
        if not sly.fs.file_exists(weights_dst_path):
            self.download(src_path=weights_url, dst_path=weights_dst_path)

        return weights_dst_path, config_path

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
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


model_dir = "my_models"  # model weights will be downloaded into this dir
settings = {"confidence_threshold": 0.7}
m = MyModel(model_dir=model_dir, custom_inference_settings=settings, use_gui=True)

m.serve()
