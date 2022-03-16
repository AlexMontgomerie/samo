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
            echo "USAGE: run_test.sh [-n (network path)] [-p (platform path)] [-o (output path)"
            echo "                   [-b (backend=finn,fpgaconvnet,hls4ml)] "
            echo "                   [-s (optimisation solver=annealing,rule,minlp,brute)] [-g]"
            echo "  -g = generate hardware"
            exit
            ;;
    esac
done

function run_hls4ml {

    # parameters
    network_path=$1
    platform_path=$2
    output_path=$3

    # run the optimiser
    time python run.py --model $network_path --backend hls4ml --platform $platform_path --output-path \
        $output_path/hls4ml.json --optimiser ${OPT} | tee outputs/saved/${network}_${N}_hls4ml.txt


    # move the output to saved outputs
    mv outputs/${network}_hls4ml.json outputs/saved/${network}_${N}_hls4ml.json

    # save the log aswell
    mv outputs/log.csv outputs/saved/${network}_${N}_hls4ml.csv

    if [ $SYNTHESIZE -eq 1 ]; then
        # build the hardware
        python -m scripts.build_hls4ml --model-path models/${network}.keras --config-path outputs/saved/${network}_${N}_hls4ml.json --platform platforms/${platform}.json --output-path outputs/hls4ml_prj | tee outputs/saved/${network}_${N}_hls4ml.txt

        # run implementation
        cd outputs/hls4ml_prj
        vivado_hls -f ../../scripts/run_hls4ml_impl.tcl myproject_prj | tee -a ../saved/${network}_${N}_hls4ml.txt
        cd ../..
    fi
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

    if [ $SYNTHESIZE -eq 1 ]; then
        # build the hardware
        $FPGACONVNET_HLS/scripts/run_network.sh -n ${network} \
            -m models/${network}.onnx \
            -p outputs/saved/${network}_${N}_fpgaconvnet.json \
            -f $part \
            -s | tee -a outputs/saved/${network}_${N}_fpgaconvnet.txt

        # save the reports
        cat partition_0/${network}_hls_prj/solution/syn/report/process_r_csynth.rpt >> outputs/saved/${network}_${N}_fpgaconvnet.txt
    fi
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

if [ $BACKEND == "fpgaconvnet"]; then
    run_fpgaconvnet
elif [ $BACKEND == "finn"]; then

elif [ $BACKEND == "hls4ml"]; then

fi

