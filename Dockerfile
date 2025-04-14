ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
RUN apt-get update

# install esp-ppq
RUN apt-get install -y git
RUN pip install git+https://github.com/espressif/esp-ppq.git

# install ultralytics for yolo
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install ultralytics
