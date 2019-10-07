# Pun detection with neural networks
### By William Grigor

## System Requirements

* An operating system running the XServer desktop environment

* Docker version 19.0 or higher

* CUDA version 10.0
    - A CUDA enabled graphics card (you can find a list of CUDA enabled GPUs [here](https://developer.nvidia.com/cuda-gpus))

* CuDNN for CUDA 10.0 (install instructions [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html))

## Installing and using

* Clone this git repository 
```bash
git clone https://github.com/FrostByte266/pun_detector.git
```

* Build the Docker container 
```bash
docker build -t puns .
```

* Run the Docker container
```bash
./container
````
OR 

```bash
xhost +
docker run -u $(id -u):$(id -g) --gpus all -it --rm --net=host -e DISPLAY -v $PWD/src:/src -v $PWD/test:/test -v $PWD/data:/data -v $PWD/logs:/logs -e TF_FORCE_GPU_ALLOW_GROWTH=true puns bash
```

* Start the main program

```bash
python /src/gui.py
```