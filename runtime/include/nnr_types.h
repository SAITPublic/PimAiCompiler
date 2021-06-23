#pragma once

#include "osal_types.h"

typedef struct __NnrBuffer {
  void* addr;                         /*!< buffer address */
  int32_t size;                       /*!< size of buffer */
} NnrBuffer;

