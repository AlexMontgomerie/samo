from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.util.basic import get_by_name

def export(network, model_path, output_path):
    if len(network.partitions) > 1:
        return 

    model = ModelWrapper(model_path)

    for finn_node in model.graph.node:
        node = finn_node.name
        layer = network.partitions[0].nodes[node]["hw"]
        finn_node = getCustomOp(finn_node)

        if get_by_name(finn_node.onnx_node.attribute, "SIMD") is not None:
            finn_node.set_nodeattr("SIMD", layer.channel_in_folding)

        if get_by_name(finn_node.onnx_node.attribute, "PE") is not None:
            finn_node.set_nodeattr("PE", layer.channel_out_folding)

    model.save(output_path)