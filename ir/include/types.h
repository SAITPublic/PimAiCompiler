/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include <float.h>
#include <unordered_map>

#include "glog/logging.h"

namespace nn_compiler
{
namespace ir
{
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
    LIST,
    TUPLE,
    IVALUE
};

enum class LayerType {
    ATENABS,
    ATENADD,
    ATENADDMM,
    ATENAND,
    ATENANY,
    ATENAPPEND,
    ATENARANGE1,
    ATENARANGE2,
    ATENARANGE3,
    ATENARGMAX,
    ATENASTENSOR,
    ATENBATCHNORM2D,
    ATENBITWISENOT,
    ATENBMM,
    ATENBOOL,
    ATENCAT,
    ATENCEIL,
    ATENCHUNK,
    ATENCLAMP,
    ATENCLEAR,
    ATENCLONE,
    ATENCONTIGUOUS,
    ATENCONV2D,
    ATENCOPY,
    ATENCPU,
    ATENCUDA,
    ATENDERIVEINDEX,
    ATENDETACH,
    ATENDIM,
    ATENDIV,
    ATENDROPOUT,
    ATENEMBEDDING,
    ATENEINSUM,
    ATENEQ,
    ATENEQUAL,
    ATENEXPAND,
    ATENFILL,
    ATENFLOORDIVIDE,
    ATENFORMAT,
    ATENFULLLIKE,
    ATENGATHER,
    ATENGE,
    ATENGETITEM,
    ATENGT,
    ATENINDEX,
    ATENINDEXPUT,
    ATENINDEXSELECT,
    ATENINT,
    ATENINTIMPLICIT,
    ATENIS,
    ATENISINF,
    ATENISNOT,
    ATENITEM,
    ATENLAYERNORM,
    ATENLEAKYRELU,
    ATENLE,
    ATENLEN,
    ATENLINEAR,
    ATENLIST,
    ATENLOG,
    ATENLOGSOFTMAX,
    ATENLSTM1,
    ATENLSTM2,
    ATENLT,
    ATENMASKEDFILL,
    ATENMASKEDSELECT,
    ATENMATMUL,
    ATENMAX,
    ATENMAXPOOL2D,
    ATENMIN,
    ATENMUL,
    ATENNE,
    ATENNEG,
    ATENNORM,
    ATENNOT,
    ATENONES,
    ATENPACKPADDEDSEQUENCE,
    ATENPADPACKEDSEQUENCE,
    ATENPOW,
    ATENRELU,
    ATENRESHAPE,
    ATENREMAINDER,
    ATENREPEAT,
    ATENSELECT,
    ATENSETITEM,
    ATENSIZE,
    ATENSLICE,
    ATENSOFTMAX,
    ATENSQUEEZE,
    ATENSUB,
    ATENSUM,
    ATENTANH,
    ATENTENSOR,
    ATENTO1,
    ATENTO2,
    ATENTOPK,
    ATENTRANSPOSE,
    ATENTRIU,
    ATENUNSQUEEZE,
    ATENVIEW,
    ATENWARN,
    ATENZEROS,
    ATENZEROSLIKE,

    PRIMBLOCK,
    PRIMCALLMETHOD,
    PRIMCONSTANT,
    PRIMDATA,
    PRIMDEVICE,
    PRIMDTYPE,
    PRIMENDIF,
    PRIMENDLOOP,
    PRIMGETATTR,
    PRIMIF,
    PRIMINPUT,
    PRIMLISTCONSTRUCT,
    PRIMLISTUNPACK,
    PRIMLOOP,
    PRIMLOOPINDEX,
    PRIMOUTPUT,
    PRIMRAISEEXCEPTION,
    PRIMSETATTR,
    PRIMTOLIST,
    PRIMTUPLECONSTRUCT,
    PRIMTUPLEINDEX,
    PRIMTUPLEUNPACK,
    PRIMTYPE,
    PRIMUNCHECKEDCAST,
    PRIMUNINITIALIZED,
    PRIMVARIABLE
};

std::string convertLayerTypeToString(LayerType type);

}  // namespace ir
}  // namespace nn_compiler
