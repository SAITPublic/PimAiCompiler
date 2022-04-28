# PIM AI Compiler

This repository contains on-device neural network compiler, GPU+PIM runtime and python libs.
1) Neural network compiler consists of IR generator, high/low level IR optimizer, PIM code generator.
2) GPU+PIM Runtime provides resource manager(model, memory) and stream executor(miopen, pim, custom kernels).
3) Python libs provide model compile and inference for both pytorch and tensorflow in runtime.


## Build Docker
* Build Image
```
$ docker/build-docker.sh
** Please comment out proxy settings in Dockerfile according to build environment ** 
```
* Launch container
```
$ docker/launch-container.sh <image name> [directory to be mapped]
```
> Please refer PIM SDK manual to configure environment prerequisites with detailed instructions.


## How to build

**Note that**

To build and test PIM AI Compiler on PIM system, environment prerequisites have to be completely setup according to PIM SDK manual.

[update submodule]   
```
$ git submodule init
$ git submodule update
```
> When your environment has to use proxy server, please ensure that the HTTP proxy settings are correct.

[set LIBTORCH_DIR]
```
$ export LIBTORCH_DIR=/home/user/.local/lib/python3.6/site-packages/torch
```

[clean build & install]
```
$ ./scripts/build.sh all -o .
```

## How to Test
### Simple example for testing PIM AI Compiler
PIM AI Compiler provides simpleMain example program for users who want to validate its functionalities.
```
./build/examples/runtime/simpleMain -h [--help]

General Options:
  -i, <input file>         Input file path
  -m, <model type>         Model type. Possible values: RNNT/GNMT/HWR
  -p, <profiling>          Run with profiling
  -?, <--help>             Help info
```

### Test Applications

[unit test]
```
$ ./build/runtime/unit-tests/NNCompilerAtenOpUnitTest
$ ./build/runtime/unit-tests/NNCompilerCustomOpUnitTest
$ ./build/runtime/unit-tests/NNCompilerPrimOpUnitTest
```

[C++ API test]
```
$ ./build/examples/runtime/simpleMain
```

[python API test]
```
$ export LIBTORCH_DIR=/home/user/.local/lib/python3.6/site-packages/torch
$ export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH
$ export PYTHONPATH=$PYTHONPATH:/opt/rocm/lib/

$ python3 ./examples/runtime/simpleMain.py
```
