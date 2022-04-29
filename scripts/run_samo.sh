#!/bin/bash
OUTPUT_PATH="outputs"
PLATFORM_PATH="platforms/zc706.json"
NETWORK_PATH="models/cnv.onnx"
BACKEND="fpgaconvnet"
OPT="rule"
SYNTH=0

# get network arguments
while getopts ":b:n:p:o:gh" opt; do
    case ${opt} in
        n ) NETWORK_PATH=$OPTARG;;
        p ) PLATFORM_PATH=$OPTARG;;
        o ) OUTPUT_PATH=$OPTARG;;
        b ) BACKEND=$OPTARG;;
        s ) OPT=$OPTARG;;
        g ) SYNTH=1;;
        h )
            echo "USAGE: run_test.sh [-n (network path)] [-p (platform path)] [-o (output path)]"
            echo "                   [-b (backend=finn,fpgaconvnet,hls4ml)] "
            echo "                   [-s (optimisation solver=annealing,rule,minlp,brute)] [-g]"
            echo "  -g = generate hardware"
            exit
            ;;
    esac
done

function run_hls4ml {

    # run the optimiser
    python -m samo --model $NETWORK_PATH --backend hls4ml --platform $PLATFORM_PATH \
        --output-path $OUTPUT_PATH/hls4ml.json --optimiser ${OPT}

    if [ $SYNTH -eq 1 ]; then
        # build the hardware
        python -m scripts.build_hls4ml --model-path models/${network}.keras \
        --config-path $OUTPUT_PATH/hls4ml.json \
        --platform $PLATFORM_PATH \
        --output-path $OUTPUT_PATH/hls4ml_prj
    fi
}

function run_fpgaconvnet {

    # get the FPGA part
    part=$( jq .part $PLATFORM_PATH )

    # run the optimiser
    python -m samo --model $NETWORK_PATH --backend fpgaconvnet \
    --platform $PLATFORM_PATH --output-path $OUTPUT_PATH --optimiser ${OPT}

}

function run_finn {
    # parameters
    network=$1
    platform=$2

    # preprocess the network
    cp models/${network}.onnx outputs/saved/finn/${network}.onnx
    cp ../finn/notebooks/samo/config/${network}.json ../finn/notebooks/samo/config.json
    jupyter nbconvert --to notebook --execute ../finn/notebooks/samo/pre_optimiser_steps.ipynb
    mv ../finn/notebooks/samo/pre_optimiser_steps.nbconvert.ipynb outputs/saved/finn/${network}_pre_optimiser_steps.nbconvert.ipynb

    # run the optimiser
    python run.py --model outputs/saved/finn/${network}_pre_optimiser.onnx --backend finn --platform platforms/${platform}.json --output-path outputs/saved/finn/${network}_post_optimiser.onnx --optimiser ${OPT} | tee outputs/saved/${network}_${N}_finn.txt

    # save the log aswell
    mv outputs/log.csv outputs/saved/${network}_${N}_finn.csv

    # build the hardware
    jupyter nbconvert --to notebook --execute ../finn/notebooks/samo/post_optimiser_steps.ipynb
    mv ../finn/notebooks/samo/post_optimiser_steps.nbconvert.ipynb outputs/saved/finn/${network}_post_optimiser_steps.nbconvert.ipynb
    cat ../finn/notebooks/samo/report.txt >> outputs/saved/${network}_${N}_finn.txt
}

if [ $BACKEND == "fpgaconvnet" ]; then
    run_fpgaconvnet
elif [ $BACKEND == "finn" ]; then
    run_finn
elif [ $BACKEND == "hls4ml" ]; then
    run_hls4ml
fi

