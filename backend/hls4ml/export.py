import json

import hls4ml
from tensorflow import keras

def export(network, model_path, output_path):

    # load keras model
    model = keras.models.load_model(model_path)

    # create the configuration for the model
    config = hls4ml.utils.config_from_keras_model(model, granularity="name")

    # set a resource-based strategy
    config["Model"]["Strategy"] = "Resource"

    # iterate over the nodes in the network
    for node in network.nodes:
        config["LayerName"][node]["ReuseFactor"] = network.nodes[node]["hw"].get_reuse_factor()

    # save the configuration file
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
