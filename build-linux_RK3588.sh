set -e

TARGET_SOC="rk3588"
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

GCC_VERSION="7.5"
GCC_COMPILER=/home/hmz/rk1.5.2/rknpu2-1.5.2/cross_compiler_7.5.0_for_firefly/bin/aarch64-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++


# build
BUILD_DIR=${ROOT_PWD}/build/GCC_7_5

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4


GCC_VERSION="10.4"
GCC_COMPILER=/home/hmz/rk1.5.2/rknpu2-1.5.2/cross_compiler_10.4.0_for_buildroot/bin/aarch64-linux

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

# build
BUILD_DIR=${ROOT_PWD}/build/GCC_10_4

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4