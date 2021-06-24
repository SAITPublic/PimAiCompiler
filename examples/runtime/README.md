there are 2 APIs C++ API and python API. so there are 2 kinds of test case.
after build, C++ example APP will located in /build/example/runtime.
and python extension lib will located in /opt/rocm/lib/Nnrt.cpython-37m-x86_64-linux-gnu.so

**1.for C++ test:**
after build, test APP in /build/example/runtime
$ cd /build/examples/runtime
$ ./simpleMain


**2. for python test:**
set /opt/rocm/lib/ in PYTHONPATH first, then run python test script.
$ export PYTHONPATH=$PYTHONPATH:/opt/rocm/lib/
$ python simpleMain.py

