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
#include "nn_runtime_api.h"

TEST(NnrUnitTest, NnrInitialize)
{
    int ret = NnrInitialize();
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, NnrDeinitialize)
{
    int ret = NnrDeinitialize();
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, NnrCompileModel)
{
    int ret = NnrCompileModel();
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, NnrPreloadModel)
{
    int ret = NnrPreloadModel();
    EXPECT_TRUE(ret == 0);
}

TEST(NnrUnitTest, NnrInferenceModel)
{
    int ret = NnrInferenceModel();
    EXPECT_TRUE(ret == 0);
}
