# Introduction

There are 2 APIs C++ API and python API. so there are 2 kinds of test case.
after build, 
* C++ example APP will located in /build/example/runtime.
* python extension lib will located in /opt/rocm/lib/Nnrt.cpython-37m-x86_64-linux-gnu.so

# How to inference

## Inference using C++ API

after build, test APP executable file are generated in /build/example/runtime

`cd /build/examples/runtime`

* ways1:

```
# set GraphIR file into ENV, the APP will load IR file based on `GRAPH_IR_FILE` ENV
export GRAPH_IR_FILE=path/to/your/graph_ir/file

# Run
./simpleMain
```

* ways2:
```
# Run
./simpleMain  path/to/your/graph_ir/file
```


## Inference using python API

The NNRuntime Python extension is installed in `/opt/rocm/lib/`, named `Nnrt.cpython-3xm-x86_64-linux-gnu.so`, so set /opt/rocm/lib/ in `PYTHONPATH` first, then run python test script.

```
export PYTHONPATH=$PYTHONPATH:/opt/rocm/lib/
python3 simpleMain.py
```



