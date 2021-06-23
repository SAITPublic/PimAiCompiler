/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <gtest/gtest.h>
#include "nn_runtime.h"

using namespace nnr;

TEST(NnrUnitTest, simpleTest)
{
    NNRuntime runtime;
    int ret = runtime.test();
    EXPECT_TRUE(ret == 0);
}


