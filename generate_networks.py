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
        layers.Dense(10, activation="relu", name="ip4")
    ], name="tfc"
)

# save the model
tfc.save("models/tfc.keras")
onnx.save(onnxmltools.convert_keras(tfc, target_opset=9), "models/tfc.onnx")

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
        layers.Dense(10, activation="relu", name="ip3")
    ], name="cnv"
)

# save the model
cnv.save("models/cnv.keras")
onnx.save(onnxmltools.convert_keras(cnv, target_opset=9), "models/cnv.onnx")

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
        layers.Dense(10, activation="relu", name="ip3")
    ], name="lenet"
)

# save the model
lenet.save("models/lenet.keras")
onnx.save(onnxmltools.convert_keras(lenet, target_opset=9), "models/lenet.onnx")

"""
simple model (from HLS4ML)
"""

simple = keras.Sequential(
    [
        layers.Dense(64, input_shape=[16], name="ip1"),
        layers.Dense(32, name="ip2"),
        layers.Dense(32, name="ip3"),
        layers.Dense(5, name="ip4")
    ], name="simple"
)

# save the model
simple.save("models/simple.keras")
onnx.save(onnxmltools.convert_keras(simple, target_opset=9), "models/simple.onnx")

