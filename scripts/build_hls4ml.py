import argparse
import json

import hls4ml
from tensorflow import keras

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
            output_dir=args.output_path,  io_type="io_stream")
            # output_dir=args.output_path, part=platform["part"],  io_type="io_stream")

    # build the hls
    hls_model.build(csim=True, cosim=True)

    # get the reports
    hls4ml.report.read_vivado_report(args.output_path)

if __name__ == "__main__":
    main()
