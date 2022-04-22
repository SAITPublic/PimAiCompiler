#!/bin/bash
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$BASE_DIR/../../
EXECUTABLE_DIR="${PROJECT_DIR}/build/examples/runtime/"
TRACE2HTML_DIR="${PROJECT_DIR}/third_party/catapult/tracing/bin/"
CURRENT_DIR=$(pwd)

function usage () {

    echo "<Usage> profiling.sh <command> <argument>

        <command>               <argument>

        --input(-i)             input file name
        --model_type(-m)        model type: RNNT/GNMT/HWR"
}

ARGS=""

for (( i=1; i<=$#; i++))
do
    case "${!i}" in
        "--ir"|"-i")
            let "i++"
            case ${!i} in
                /*) IR=${!i};;
                *)  IR=$(pwd)/${!i};;
            esac
            ARGS="$ARGS -i $(realpath $IR)"
            ;;
        "--model_type"|"-m")
            let "i++"
            m=${!i}
            ARGS="$ARGS -m $(echo ${!i})"
            ;;
        "--help"|"-h")
            usage
            exit 1
            ;;
       *)
            ARGS="$ARGS ${!i}"
    esac
done

ARGS="$ARGS -p 1"

if [[ $IR == "" ]]
then
    usage
    exit 1
elif [[ ! -e $IR ]]
then
    echo "input file does not exist"
    exit 1
fi

if [[ $m == "" ]]
then
    usage
    exit 1
fi

$EXECUTABLE_DIR/simpleMain $ARGS
echo $ARGS
${TRACE2HTML_DIR}trace2html ${CURRENT_DIR}/trace.json --output=${m}_Profiling.html
