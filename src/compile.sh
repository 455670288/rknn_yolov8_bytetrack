g++ -g -o yolov8_p2_test main_MultiDetect.cpp ItcMultiDetect.cc ItcMultiDetectImpl.cc postprocess.cc resize_function.cc rknn_utils.cc coreNum.cc -L/home/firefly/ljh/ItcMultiDetect_yolov8_sin/libs/librknn_api/lib -lrknnrt -I/home/firefly/ljh/ItcMultiDetect_yolov8_sin/libs/librknn_api/include -L/home/firefly/ljh/ItcMultiDetect_yolov8_sin/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/lib -lopencv_world -I/home/firefly/ljh/ItcMultiDetect_yolov8_sin/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/include -I/home/firefly/ljh/ItcMultiDetect_yolov8_sin/include -I/home/firefly/ljh/ItcMultiDetect_yolov8_sin/src -pthread


#添加rknn opencv路径到环境变量
export LD_LIBRARY_PATH=/home/firefly/ljh/ItcMultiDetect/libs/librknn_api/lib:/home/firefly/ljh/ItcMultiDetect/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/firefly/ljh/ItcMultiDetect/libs/librknn_api/lib:/home/firefly/ljh/ItcMultiDetect/3rdparty/GCC_7_5/opencv/opencv-linux-aarch64/lib:$LD_LIBRARY_PATH



