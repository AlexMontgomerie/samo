import onnx
import onnxmltools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
tfc model (from FINN)
"""

tfc = keras.Sequential(
    [
        layers.Dense(64, input_shape=[784], activation="relu", name="ip1"),
        layers.Dense(64, activation="relu", name="ip2"),
        layers.Dense(64, activation="relu", name="ip3"),
        layers.Dense(10, name="ip4")
    ], name="tfc"
)

# save the model
tfc.save("models/tfc.keras")
onnx.save(onnxmltools.convert_keras(tfc, target_opset=9), "models/tfc.onnx")

"""
sfc model (from FINN)
"""

sfc = keras.Sequential(
    [
        layers.Dense(256, input_shape=[784], activation="relu", name="ip1"),
        layers.Dense(256, activation="relu", name="ip2"),
        layers.Dense(256, activation="relu", name="ip3"),
        layers.Dense(10, name="ip4")
    ], name="sfc"
)

# save the model
sfc.save("models/sfc.keras")
onnx.save(onnxmltools.convert_keras(sfc, target_opset=9), "models/sfc.onnx")

"""
lfc model (from FINN)
"""

lfc = keras.Sequential(
    [
        layers.Dense(1024, input_shape=[784], activation="relu", name="ip1"),
        layers.Dense(1024, activation="relu", name="ip2"),
        layers.Dense(1024, activation="relu", name="ip3"),
        layers.Dense(10, name="ip4")
    ], name="lfc"
)

# save the model
lfc.save("models/lfc.keras")
onnx.save(onnxmltools.convert_keras(lfc, target_opset=9), "models/lfc.onnx")

"""
cnv model (from FINN)
"""

cnv = keras.Sequential(
    [
        layers.Conv2D(64, 3, input_shape=[32,32,3], activation="relu", name="conv1"),
        layers.Conv2D(64, 3, activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),
        layers.Conv2D(128, 3, activation="relu", name="conv3"),
        layers.Conv2D(128, 3, activation="relu", name="conv4"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),
        layers.Conv2D(256, 3, activation="relu", name="conv5"),
        layers.Conv2D(256, 3, activation="relu", name="conv6"),
        layers.Flatten(),
        layers.Dense(512, activation="relu", name="ip1"),
        layers.Dense(512, activation="relu", name="ip2"),
        layers.Dense(10, name="ip3")
    ], name="cnv"
)

# save the model
cnv.save("models/cnv.keras")
onnx.save(onnxmltools.convert_keras(cnv, target_opset=9, channel_first_inputs=["conv1_input"]), "models/cnv.onnx")

"""
lenet model (from FPGAConvNet)
"""

lenet = keras.Sequential(
    [
        layers.Conv2D(20, 5, input_shape=[28,28,1], name="conv1"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),
        layers.Conv2D(50, 5, name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),
        layers.Flatten(),
        layers.Dense(500, activation="relu", name="ip1"),
        layers.Dense(10, name="ip3")
    ], name="lenet"
)

# save the model
lenet.save("models/lenet.keras")
onnx.save(onnxmltools.convert_keras(lenet, target_opset=9, channel_first_inputs=["conv1_input"]), "models/lenet.onnx")

"""
mpcnn (from FPGAConvNet)
"""

mpcnn = keras.Sequential(
    [
        layers.Conv2D(20, 5, input_shape=[32, 32, 1], name="conv1"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),
        layers.Conv2D(20, 5, name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),
        layers.Conv2D(20, 3, name="conv3"),
        layers.Flatten(),
        layers.Dense(300, name="ip1"),
        layers.Dense(6, name="ip2")
    ]
)

# save the model
mpcnn.save("models/mpcnn.keras")
onnx.save(onnxmltools.convert_keras(mpcnn, target_opset=9, channel_first_inputs=["conv1_input"]), "models/mpcnn.onnx")

"""
simple model (from HLS4ML)
"""

simple = keras.Sequential(
    [
        layers.Dense(64, input_shape=[16], activation="relu", name="ip1"),
        layers.Dense(32, activation="relu", name="ip2"),
        layers.Dense(32, activation="relu", name="ip3"),
        layers.Dense(5, name="ip4")
    ], name="simple"
)

# save the model
simple.save("models/simple.keras")
onnx.save(onnxmltools.convert_keras(simple, target_opset=9), "models/simple.onnx")

"""
AlexNet
"""

alexnet = keras.Sequential(
    [
        layers.ZeroPadding2D(padding=(2,2), input_shape=[224, 224, 3]),
        layers.Conv2D(64, 11, strides=(4,4), activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool1"),

        layers.Conv2D(192, 5, padding='same', activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool2"),

        layers.Conv2D(384, 3, padding='same', activation="relu", name="conv3"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv4"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv5"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool3"),

        layers.Flatten(),
        layers.Dense(4096, name="ip1"),
        layers.Dense(4096, name="ip2"),
        layers.Dense(1000, name="ip3")
    ], name="alexnet"
)

# save the model
alexnet.save("models/alexnet.keras")
onnx.save(onnxmltools.convert_keras(alexnet, target_opset=9,channel_first_inputs=['zero_padding2d_input']), "models/alexnet.onnx")

"""
AlexNet (fpgaconvnet)
"""

alexnet_fpgaconvnet = keras.Sequential(
    [
        # layers.ZeroPadding2D(padding=(2,2), input_shape=[224, 224, 3]),
        layers.Conv2D(64, 11, strides=(4,4), input_shape=[228, 228, 3],
            activation="relu", padding="valid", name="conv1"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool1"),

        layers.Conv2D(192, 5, padding='same', activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool2"),

        layers.Conv2D(384, 3, padding='same', activation="relu", name="conv3"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv4"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv5"),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool3"),
        layers.Flatten()
    ], name="alexnet_fpgaconvnet"
)

# save the model
alexnet_fpgaconvnet.save("models/alexnet_fpgaconvnet.keras")
onnx.save(onnxmltools.convert_keras(alexnet_fpgaconvnet, target_opset=9,channel_first_inputs=['conv1_input']), "models/alexnet_fpgaconvnet.onnx")

"""
AlexNet-MP2
"""

alexnet_mp2 = keras.Sequential(
    [
        layers.ZeroPadding2D(padding=(4,4), input_shape=[224, 224, 3]),
        layers.Conv2D(64, 11, strides=(4,4), activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),

        layers.Conv2D(192, 5, padding='same', activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),

        layers.Conv2D(384, 3, padding='same', activation="relu", name="conv3"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv4"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv5"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool3"),

        layers.Flatten(),
        layers.Dense(4096, name="ip1"),
        layers.Dense(4096, name="ip2"),
        layers.Dense(1000, name="ip3")
    ], name="alexnet_mp2"
)

# save the model
alexnet_mp2.save("models/alexnet_mp2.keras")
onnx.save(onnxmltools.convert_keras(alexnet_mp2, target_opset=9,channel_first_inputs=['zero_padding2d_input']), "models/alexnet_mp2.onnx")

"""
VGG-11
"""

vgg11 = keras.Sequential(
    [
        layers.Conv2D(64, 3, padding='same', input_shape=[224, 224, 3], activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),

        layers.Conv2D(128, 3, padding='same', activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),

        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv3"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv4"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool3"),

        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv5"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv6"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool4"),

        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv7"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv8"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool5"),

        layers.Flatten(),
        layers.Dense(4096, name="ip1"),
        layers.Dense(4096, name="ip2"),
        layers.Dense(1000, name="ip3")
    ], name="vgg11"
)

# save the model
vgg11.save("models/vgg11.keras")
onnx.save(onnxmltools.convert_keras(vgg11, target_opset=9,channel_first_inputs=['conv1_input']), "models/vgg11.onnx")

"""
VGG-16 (fpgaconvnet)
"""

vgg16_fpgaconvnet = keras.Sequential(
    [
        layers.Conv2D(64, 3, padding='same', input_shape=[224, 224, 3], activation="relu", name="conv1"),
        layers.Conv2D(64, 3, padding='same', activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1"),

        layers.Conv2D(128, 3, padding='same', activation="relu", name="conv3"),
        layers.Conv2D(128, 3, padding='same', activation="relu", name="conv4"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2"),

        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv5"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv6"),
        layers.Conv2D(256, 3, padding='same', activation="relu", name="conv7"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool3"),

        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv8"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv9"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv10"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool4"),

        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv11"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv12"),
        layers.Conv2D(512, 3, padding='same', activation="relu", name="conv13"),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool5"),
        layers.Flatten()

    ], name="vgg16_fpgaconvnet"
)

# save the model
vgg16_fpgaconvnet.save("models/vgg16_fpgaconvnet.keras")
onnx.save(onnxmltools.convert_keras(vgg16_fpgaconvnet, target_opset=9,channel_first_inputs=['conv1_input']), "models/vgg16_fpgaconvnet.onnx")

"""
MobileNet-V1
"""

mobilenetv1 = keras.Sequential(
    [
        layers.Conv2D(32, 3, strides=(2,2), padding='same', input_shape=[224, 224, 3], activation="relu", use_bias=False, name="conv1"),

        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv2"),
        layers.Conv2D(64, 1, padding='same', activation="relu", use_bias=False, name="conv3"),

        layers.DepthwiseConv2D(3, strides=(2,2), padding='same', activation="relu", use_bias=False, name="conv4"),
        layers.Conv2D(128, 1, padding='same', activation="relu", use_bias=False, name="conv5"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv6"),
        layers.Conv2D(128, 1, padding='same', activation="relu", use_bias=False, name="conv7"),

        layers.DepthwiseConv2D(3, strides=(2,2), padding='same', activation="relu", use_bias=False, name="conv8"),
        layers.Conv2D(256, 1, padding='same', activation="relu", use_bias=False, name="conv9"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv10"),
        layers.Conv2D(256, 1, padding='same', activation="relu", use_bias=False, name="conv11"),

        layers.DepthwiseConv2D(3, strides=(2,2), padding='same', activation="relu", use_bias=False, name="conv12"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv13"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv14"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv15"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv16"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv17"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv18"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv19"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv20"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv21"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv22"),
        layers.Conv2D(512, 1, padding='same', activation="relu", use_bias=False, name="conv23"),

        layers.DepthwiseConv2D(3, strides=(2,2), padding='same', activation="relu", use_bias=False, name="conv24"),
        layers.Conv2D(1024, 1, padding='same', activation="relu", use_bias=False, name="conv25"),
        layers.DepthwiseConv2D(3, padding='same', activation="relu", use_bias=False, name="conv26"),
        layers.Conv2D(1024, 1, padding='same', activation="relu", use_bias=False, name="conv27"),

        layers.AvgPool2D(pool_size=(7,7), name="pool1"),
        layers.Flatten(),
        layers.Dense(1000, name="ip1")
    ], name="mobilenetv1"
)


# save the model
mobilenetv1.save("models/mobilenetv1.keras")
onnx.save(onnxmltools.convert_keras(mobilenetv1, target_opset=9,channel_first_inputs=['conv1_input']), "models/mobilenetv1.onnx")
