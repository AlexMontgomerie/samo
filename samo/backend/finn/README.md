To run optimiser with finn:

FINN is not actually supported as a submodule of the optimiser. Clone finn as a full git repo and store it outside the optimiser folder.


Start the docker

Edit SAME_DIR to specify the optimiser path

Edit FINN_HOST_BUILD_DIR to change hls build location (default in /tmp)

```
bash run-docker.sh
```

```
cd ../same
bash run.sh
```
