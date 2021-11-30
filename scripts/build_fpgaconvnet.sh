#!/bin/bash

name="simple"
model_path="models/simple.onnx"
hardware_path="outputs/simple_fpgaconvnet.json"

$FPGACONVNET_HLS/scripts/run_network.sh -n $name \
    -m $model_path \
    -p $hardware_path \
    -b xilinx.com:zc706:part0:1.4 \
    -f xcku115-flvb2104-2-i \
    -i
