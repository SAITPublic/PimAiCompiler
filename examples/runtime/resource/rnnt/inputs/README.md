The RNNT model has 2 inputs:
* feature_len.bin   binary file, dtype=int64, shape=[1], value=341
* feature.bin       binary file, dtype=float16, shape=[341, 1, 240]

These are raw pytorch file, it can't load directly in C++
* feature_len.pth:   pytorch file
* feature.pth:       pytorch file
