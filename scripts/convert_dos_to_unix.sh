#!/bin/bash

find runtime    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.fimk' | xargs dos2unix
find examples   -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs dos2unix
find tools      -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs dos2unix
find tf_fim_ops -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs dos2unix
find external_libs/include/dramsim2 -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs dos2unix
