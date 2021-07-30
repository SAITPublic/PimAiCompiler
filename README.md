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
if you do not have libtorch, you can get it from 75.12.84.67:/home/srcxfim/share/torch-1.7.0a0-cp36-cp36m-linux_x86_64.whl
and install the torch in container first
```
$ pip install torch-1.7.0a0-cp36-cp36m-linux_x86_64.whl

```

**modify Caffe2Targets.cmake**, while `find_package(Torch REQUIRED)` can't find **gloo_hip** 

```
$ cd $LIBTORCH_DIR/share/cmake/Caffe2

# delete **gloo_hip**
Caffe2Targets.cmake 93: INTERFACE_LINK_LIBRARIES c10_hip;torch_cpu_library; ... gloo_hip" 
Caffe2Targets.cmake 161: foreach(_target "protobuf::libprotobuf" "gloo_hip" )

```

[clean build & install]
```
$ ./scripts/build.sh all -o .
```
# How to run

## Compiler
### Set ME_PASS_CONFIG_PATH
```
default file is: compiler/include/middlend/passes/pass_config.json
```

### Run compiler
```
./build/compiler/compiler -h [--help]

General Options:
  -h [ --help ]            Help info
  -i, <input file>         Input file path
  -l, <compile level>      Compile level. Possible values (default: 0):
                                                    0 (frontend->middlend->backend);
                                                    1 (middlend->backend);
                                                    2 (backend)
  -g, <graphgen_path>      GraphGen real path
  -c, <configuration file> [middlend] passes configuration file path. default: compiler/include/middlend/passes/pass_config.json
```
### Note:
When running with compile level 0 (frontend->middlend->backend) in __docker__, please update docker with:
```
sudo apt-get update
sudo apt-get install libprotobuf-dev protobuf-compiler --fix-missing
```

And then __rebuild GraphGen in docker__.

# How to test

[unit test]
```
$ ./build/runtime/unit-tests/NnrtUnitTest
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

