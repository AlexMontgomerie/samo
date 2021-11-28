#!/bin/bash
N=7
OPT="init"

function run_hls4ml {

    # parameters
    network=$1
    platform=$2

    # run the optimiser
    # python run.py --model models/${network}.keras --backend hls4ml --platform platforms/${platform}.json --output-path outputs/${network}_hls4ml.json --optimiser annealing
    python run.py --model models/${network}.keras --backend hls4ml --platform platforms/${platform}.json --output-path outputs/${network}_hls4ml.json --optimiser ${OPT}

    # move the output to saved outputs
    mv outputs/${network}_hls4ml.json outputs/saved/${network}_${N}_hls4ml.json

    # save the log aswell
    mv outputs/log.csv outputs/saved/${network}_${N}_hls4ml.csv

    # build the hardware
    python -m scripts.build_hls4ml --model-path models/${network}.keras --config-path outputs/saved/${network}_${N}_hls4ml.json --platform platforms/${platform}.json --output-path outputs/hls4ml_prj | tee outputs/saved/${network}_${N}_hls4ml.txt

    # run implementation
    cd outputs/hls4ml_prj
    vivado_hls -f ../../scripts/run_hls4ml_impl.tcl myproject_prj | tee -a ../saved/${network}_${N}_hls4ml.txt
    cd ../..
}

function run_fpgaconvnet {

    # parameters
    network=$1
    platform=$2
    part=$( jq .part platforms/${platform}.json )

    # run the optimiser
    python run.py --model models/${network}.onnx --backend fpgaconvnet --platform platforms/${platform}.json --output-path outputs/ --optimiser ${OPT} | tee outputs/saved/${network}_${N}_fpgaconvnet.txt

    # move the output to saved outputs
    mv outputs/same.json outputs/saved/${network}_${N}_fpgaconvnet.json

    # save the log aswell
    mv outputs/log.csv outputs/saved/${network}_${N}_fpgaconvnet.csv

    # clean the build directory
    rm -rf partition_0/*

    # build the hardware
    $FPGACONVNET_HLS/scripts/run_network.sh -n ${network} \
        -m models/${network}.onnx \
        -p outputs/saved/${network}_${N}_fpgaconvnet.json \
        -f $part \
        -s | tee -a outputs/saved/${network}_${N}_fpgaconvnet.txt

    # save the reports
    cat partition_0/${network}_hls_prj/solution/syn/report/process_r_csynth.rpt >> outputs/saved/${network}_${N}_fpgaconvnet.txt

}

# HLS4ML
run_hls4ml simple zedboard
run_hls4ml tfc zedboard
# run_hls4ml lenet zc706
# run_hls4ml cnv u250

# fpgaconvnet
run_fpgaconvnet simple zedboard
run_fpgaconvnet tfc zedboard
run_fpgaconvnet lenet zc706
run_fpgaconvnet cnv u250
