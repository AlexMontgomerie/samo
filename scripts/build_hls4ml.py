import shutil
import argparse
import json

import hls4ml
from tensorflow import keras

from backend.hls4ml.parser import parse

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="run hls4ml configuration")
    parser.add_argument("-m", "--model-path", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras)")
    parser.add_argument("-c", "--config-path", metavar="PATH", required=True,
            help="configuration path (.json)")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for HLS implementation")
    args = parser.parse_args()

    # clean any existing project
    shutil.rmtree(args.output_path, ignore_errors=True)

    # load the configuration
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # load the platform path
    with open(args.platform, "r") as f:
        platform = json.load(f)

    # load the keras model
    model = keras.models.load_model(args.model_path)

    # create the hls model
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config,
            output_dir=args.output_path,  io_type="io_stream", fpga_part=platform["part"])

    # parse the model for same
    net = parse(args.model_path)

    #update network kernel folding:
    for node in config["LayerName"]:
        if node in net.nodes:
            net.nodes[node]["hw"].kernel_folding = config["LayerName"][node]["ReuseFactor"]
            print(net.nodes[node]["hw"].kernel_folding)
    # report the estimates
    net.summary()

    # # build the hls
    hls_model.build(csim=True, cosim=True)

    # get the reports
    hls4ml.report.read_vivado_report(args.output_path)

if __name__ == "__main__":
    main()
