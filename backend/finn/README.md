To run optimiser with finn:

FINN is not actually supported as a submodule of the optimiser. Clone finn as a full git repo and store it outside the optimiser folder.


Start the docker

Edit SAME_DIR to specify the optimiser path

Edit FINN_HOST_BUILD_DIR to change hls build location (default in /tmp)

```
bash run-docker.sh
```

Prepare the model
```
jupyter nbconvert --to notebook --execute "notebooks/same/pre_optimiser_steps.ipynb"
```

Run the optimiser
```
cd ../same
python run.py -m "../finn/hls4ml_example_pre_optimiser.onnx" -b "finn" --p "platforms/zedboard.json" -o "../finn/hls4ml_example_post_optimiser.onnx" 
```

Generate the hardware
```
cd ../finn
jupyter nbconvert --to notebook --execute "notebooks/same/post_optimiser_steps.ipynb"
```
