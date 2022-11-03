# Introduction

There are two test examples provided with C++ API and Python API repectively,
which are designed to validate the functionality of pipeline.

After successful build,
* C++ example APP will be located in /build/example/runtime/.
* Python extension lib will be located in /opt/rocm/lib/.

---

# How to run

## Inference with C++ API

```
<Usage> ./build/examples/runtime/simpleMain -h [--help]

  -i, <input file>         Input file path
  -m, <model type>         Model type. Possible values: RNNT/GNMT/HWR/Transformer/SwitchTransformer
  -p, <profiling>          Run with profiling
  -?, <--help>             Help info
```


## Inference with python API
```
$ export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH
$ export PYTHONPATH=$PYTHONPATH:/opt/rocm/lib/
```

```
<Usage> python3 example/runtime/python/simpleMain.py

  --input_file         Input file path
  --model_kind         Supported model type: RNNT/GNMT/HWR/Transformer/SwitchTransformer

```
