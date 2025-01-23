cmake_minimum_required(VERSION 3.4.1)

project(Detect)

# 指定编译器进行交叉编译
#set(CMAKE_C_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-gcc)
#set(CMAKE_CXX_COMPILER /home/ljh/ljh_nfs/my_toolchain/host/bin/aarch64-buildroot-linux-gnu-g++)


# set(CMAKE_BUILD_TYPE "Debug") #启用调试信息
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置链接器标志以允许共享库中的未定义符
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/Detect_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# set(LIB_ARCH "x86_64")
set(LIB_ARCH aarch64)
if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()
include_directories(${CMAKE_SOURCE_DIR})

# rknn api
#指明目标芯片
set(TARGET_SOC "rk3588" CACHE STRING "Target SoC type") 
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/libs/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/libs/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588 or rv110x")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    if (GCC_VERSION STREQUAL "7.5")
      set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/share/OpenCV)
    else()
      set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/GCC_10_4/opencv/opencv-linux-aarch64/share/OpenCV) #链接静态库
    endif()
  endif()
endif()
find_package(OpenCV REQUIRED) #在OpenCV_DIR路径中根据OpenCV包名寻找OpenCV库的配置文件


set(CMAKE_INSTALL_RPATH "lib")

include_directories( ${CMAKE_SOURCE_DIR}/include)

include_directories( ${CMAKE_SOURCE_DIR}/src)


# 可执行文件
add_executable(Detect 
               src/main_MultiDetect.cpp
               src/Detect.cc
               src/resize_function.cc
               src/rknn_utils.cc
               src/postprocess.cc
               src/DetectImpl.cc
               src/BYTETracker.cpp
               src/kalmanFilter.cpp
               src/lapjv.cpp
               src/STrack.cpp
               src/utils.cpp
               )

#将库链接到可执行文件               
target_link_libraries(Detect
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

#链接phtead库
find_package(Threads REQUIRED)
target_link_libraries(Detect ${CMAKE_THREAD_LIBS_INIT})
