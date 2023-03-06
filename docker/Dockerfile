FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y ffmpeg libsm6 libxext6  
RUN apt-get install ninja-build
RUN python3 -m pip install "git+https://github.com/facebookresearch/detectron2.git"
RUN python3 -m pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN python3 -m pip install supervisely==6.69.66