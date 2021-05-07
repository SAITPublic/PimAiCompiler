# NNCompiler

This repository contains on-device neural network compiler, GPU+PIM runtime and python libs.
1) Neural network compiler consists of IR generator, high/low level IR optimizer, PIM code generator.
2) GPU+PIM Runtime provides resource manager(model, memory) and stream executor(miopen, pim, custom kernels).
3) Python libs provide model compile and inference for both pytorch and tensorflow in runtime.

# Build Docker
* Build Image
```
$ docker/build-docker.sh
** In case building docker outside SAIT servers, comment out proxy settings in Dockerfile ** 
```
* Launch container
```
launch-container.sh <image name> [directory to be mapped]
```


# How to build

[clean build & install]

./script/build.sh all -o .

# How to test

[unit test]

./build/runtime/unit-tests/NnrUnitTest
