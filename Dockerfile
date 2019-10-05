FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk
RUN pip install --upgrade numpy==1.16.4

