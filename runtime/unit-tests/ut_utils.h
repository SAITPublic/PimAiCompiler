/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include <torch/script.h>
#include <string>

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

#define ASSERT_ALLCLOSE(t1, t2)       \
    ASSERT_TRUE(t1.is_same_size(t2)); \
    ASSERT_TRUE(t1.allclose(t2));

// https://pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose
#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
    ASSERT_TRUE(t1.is_same_size(t2));                  \
    ASSERT_TRUE(t1.allclose(t2, atol, rtol));
