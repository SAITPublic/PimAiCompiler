#pragma once

#include "osal_types.h"

#define __NNR_API__

typedef struct __NnrtBuffer {
  void* addr;                         /*!< buffer address */
  int32_t size;                       /*!< size of buffer */
} NnrtBuffer;

