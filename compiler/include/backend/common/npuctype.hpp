/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

namespace nn_compiler {
/**
 * @brief. NPUC Buffer struct
 * @details. This is used to pass input data (bias, kernel, and ncp binary)
 */
typedef struct __NCPBuffer {
    unsigned char* addr = 0; /* buffer address */
    unsigned int   size = 0; /* buffer size */
} NCPBuffer;

/**
 * @brief Vector of strings in the form of: <option>[=<value>] for setting options in passes configuration file
 */
using PassConfOptions = std::vector<std::string>;

} // namespace nn_compiler
