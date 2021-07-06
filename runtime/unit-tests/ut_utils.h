#pragma once

#include<torch/script.h>
#include<string>

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

#define ASSERT_ALLCLOSE(t1, t2)       \
    ASSERT_TRUE(t1.is_same_size(t2)); \
    ASSERT_TRUE(t1.allclose(t2));

#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
    ASSERT_TRUE(t1.is_same_size(t2));                  \
    ASSERT_TRUE(t1.allclose(t2, atol, rtol));
