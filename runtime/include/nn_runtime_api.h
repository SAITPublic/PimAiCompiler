/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _NN_RUNTIME_API_H_
#define _NN_RUNTIME_API_H_

#define __NNR_API__

__NNR_API__ int NnrTest(void);
__NNR_API__ int NnrInitialize(void);
__NNR_API__ int NnrDeinitialize(void);
__NNR_API__ int NnrCompileModel(void);
__NNR_API__ int NnrPreloadModel(void);
__NNR_API__ int NnrInferenceModel(void);

#endif /* _NN_RUNTIME_API_H_ */
