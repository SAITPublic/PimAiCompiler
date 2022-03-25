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

Ensure that the HTTP proxy settings are correct.

[update submodule]   
```
$ git submodule init    
$ git submodule update    
```

[set LIBTORCH_DIR]
```
$ export LIBTORCH_DIR=/home/user/.local/lib/python3.6/site-packages/torch
```
if you do not have libtorch, you can get it from 75.12.84.95: /home/srcxfim/public_file/torch-1.10.0a0+git6f4069a-cp36-cp36m-linux_x86_64.whl
and install the torch package in container first
```
$ pip install torch-1.10.0a0+git6f4069a-cp36-cp36m-linux_x86_64.whl

```

[clean build & install]
```
$ ./scripts/build.sh all -o .
```

# How to run

## NNCompiler

```
./build/examples/runtime/simpleMain -h [--help]

General Options:
  -i, <input file>         Input file path
  -m, <model type>         Model type. Possible values: RNNT/GNMT/HWR
  -p, <profiling>          Run with profiling
  -?, <--help>             Help info
```

# How to test

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

