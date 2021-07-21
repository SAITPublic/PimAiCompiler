
option(BUILD_PY_RUNTIME "option for build py_nnruntime extension" ON)
option(BUILD_RUNTIME_UTEST "option for build utest" OFF)

add_subdirectory(source)

if(BUILD_PY_RUNTIME)
    add_subdirectory(python)
endif(BUILD_PY_RUNTIME)

if(BUILD_RUNTIME_UTEST)
    add_subdirectory(unit-tests)
endif(BUILD_RUNTIME_UTEST)
