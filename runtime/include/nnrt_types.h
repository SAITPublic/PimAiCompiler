#pragma once

#include "osal_types.h"

#define __NNRT_API__

typedef struct __NnrtBuffer {
    void* addr;   /*!< buffer address */
    int32_t size; /*!< size of buffer */
} NnrtBuffer;

// these dtype are same with GraphGen
enum DataType {
    UNDEFINED = 0,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    INT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BOOL,
    STRING,
    DEVICE,
    TENSOR,
    NONE,
    LIST
};
