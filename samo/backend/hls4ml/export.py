import json

import hls4ml
from tensorflow import keras

def export(network, model_path, output_path):

    # load keras reference model
    ref_model = keras.models.load_model(model_path)
    ref_model.summary()

    # create the reference configuration for the model
    ref_config = hls4ml.utils.config_from_keras_model(ref_model, granularity="name")

    # set a resource-based strategy
    ref_config["Model"]["Strategy"] = "Resource"

    # iterate over partitions
    for i, partition in enumerate(network.partitions):

        # create a new config for the partition
        config = {"LayerName":{}}

        # copy model info over
        config["Model"] = ref_config["Model"]

        # iterate over the nodes in the partition
        for node in partition.nodes:
            # copy node layer info from reference
            config["LayerName"][node] = ref_config["LayerName"][node]
            # update reuse factor
            config["LayerName"][node]["ReuseFactor"] = partition.nodes[node]["hw"].get_reuse_factor()

        # save the configuration file
        with open(f"{output_path}_{i}.json", "w") as f:
            json.dump(config, f, indent=4)
