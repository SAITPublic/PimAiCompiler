#!/bin/bash
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$BASE_DIR/../../
MOUNT_VOLUME=""

# SCHEMA
SCHEMA_DEFAULT_DIR=$(realpath $PROJECT_DIR/../npu_ir)
SCHEMA_GEN_DIR="$PROJECT_DIR/build/python"
SCHEMA_DIR=""

function usage () {

    echo "<Usage> visualizer.sh <command> <argument>

        <command>           <argument>

        --ir(-i)            input ir file name
        --rank-dir(-rd)     Graph Direction Left-Right / Top-Bottom"
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
        "--schema"|"-s")
            let "i++"
            case ${!i} in
                /*) SCHEMA_DIR=${!i};;
                *)  SCHEMA_DIR=$(pwd)/${!i};;
            esac
            SCHEMA_DIR=$(realpath $SCHEMA_DIR)
            SCHEMA_GEN_ARG="-s $SCHEMA_DIR"
            ;;
        "--help"|"-h")
            usage
            exit 1
            ;;
       *)
            ARGS="$ARGS ${!i}"
    esac
done

if [[ $IR == "" ]]
then
    usage
    exit 1
elif [[ ! -e $IR ]]
then
    echo "input IR does not exist"
    exit 1
fi

if [[ "$(ls -A $SCHEMA_GEN_DIR 2> /dev/null)" == "" ]]
then
    $PROJECT_DIR/scripts/build/local_build.sh $SCHEMA_GEN_ARG --only_schema
    if [[ $? -ne 0 ]]
    then
        echo "Error in Flatbuffers schema conversion"
        exit 1
    fi
fi

PYTHONPATH=$SCHEMA_GEN_DIR:$PYTHONPATH python3 $PROJECT_DIR/tools/ir_visualizer/ir_visualizer.py $ARGS
