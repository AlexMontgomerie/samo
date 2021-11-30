# SAMO: Streaming Architecture Mapping Optimiser

The SAMO framework provides a method of optimising the mapping of a Convolutional Neural Network model onto an FPGA platform for Streaming Architecture frameworks. We currently support the following frameworks:

- [FINN](https://github.com/Xilinx/finn)
- [HLS4ML](https://github.com/fastmachinelearning/hls4ml)
- [fpgaConvNet](https://github.com/AlexMontgomerie/fpgaconvnet-optimiser)

Currently both a Simulated Annealing and Brute Force optimiser are implemented.

## Usage

```
>> python -m samo -h
usage: __main__.py [-h] -m PATH -b {fpgaconvnet,finn,hls4ml} -p PATH -o PATH
                   [--optimiser {brute,annealing,init}]

SAMO CNN optimiser

optional arguments:
  -h, --help            show this help message and exit
  -m PATH, --model PATH
                        path to the CNN model that you wish to optimise
                        (.keras, .onnx)
  -b {fpgaconvnet,finn,hls4ml}, --backend {fpgaconvnet,finn,hls4ml}
                        target backend for accelerating the model
  -p PATH, --platform PATH
                        hardware platform details (.json)
  -o PATH, --output-path PATH
                        output path for the optimised model (.json, .onnx)
  --optimiser {brute,annealing,init}
                        optimiser to use
```

### FINN

> TODO

### HLS4ML


