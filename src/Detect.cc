#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "Detect.h"
#include "DetectImpl.h"
#include "resize_function.h"
#include "rknn_utils.h"
#include "postprocess.h"
#include "timer.h"

Detect::Detect()
{
    pImpl = new Impl();
}

Detect::~Detect()
{
    delete pImpl;
}

int Detect::init_model(const char *model_path, const float &nmsThreshold, const float &boxThreshold, const int &NPUcore, const std::vector<float> &confs)
{
    return pImpl->init_model(model_path, nmsThreshold, boxThreshold, NPUcore, confs);
}

int Detect::detect(const cv::Mat &image, std::vector<int> *ids, std::vector<float> *conf, std::vector<cv::Rect> *boxes)
{
    return pImpl->detect(image, ids, conf, boxes);
}