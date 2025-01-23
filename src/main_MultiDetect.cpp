/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include<fstream>
#include <unordered_set>
#include <thread>
#include <queue>  
#include <mutex>  
#include <condition_variable>
#include <atomic>

#include "Detect.h"
#include "postprocess.h"
#include "timer.h"
#include "BYTETracker.h"

using namespace std;






double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

const std::vector<std::string> names = {"person", "head_helmet","head","reflective_clothes","smoking","calling","falling","face_mask","car","bicycle","motorcycle",
                    "fumes","fire","head_hat","normal_clothes","face","play_phone","other","knife"};

const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // 蓝色
        cv::Scalar(0, 255, 0),    // 绿色
        cv::Scalar(0, 0, 255),    // 红色
        cv::Scalar(255, 255, 0),  // 青色
        cv::Scalar(255, 0, 255),  // 品红色
        cv::Scalar(0, 255, 255),  // 黄色
        cv::Scalar(192, 192, 192), // 浅灰色
        cv::Scalar(128, 0, 0),    // 深红色
        cv::Scalar(128, 128, 0),  // 橄榄色
        cv::Scalar(0, 128, 0),    // 深绿色
        cv::Scalar(128, 0, 128),  // 紫色
        cv::Scalar(0, 128, 128),  // 深青色
        cv::Scalar(0, 0, 128),    // 深蓝色
        cv::Scalar(255, 128, 0),  // 橙色
        cv::Scalar(255, 0, 128),  // 红紫色
        cv::Scalar(128, 255, 0),  // 黄绿色
        cv::Scalar(0, 255, 128),  // 青绿色
        cv::Scalar(128, 0, 255),  // 蓝紫色
        cv::Scalar(255, 255, 255) // 白色
    };

const float person_thres = 0.2;      //0
const float head_helmet_thres = 0.15;        //1
const float head_thres = 0.15;                //2
const float reflective_clothes_thres = 0.5;   //3
const float smoking_thres = 0.15;    //4
const float calling_thres = 0.15;     //5
const float falling_thres = 0.5;       //6
const float face_mask_thres = 0.5;     //7
const float car_thres = 0.5;           //8
const float bicycle_thres = 0.5;        //9
const float motorcycle_thres = 0.5;     //10
const float fumes_thres = 0.5;          //11
const float fire_thres = 0.5;          //12
const float head_hat_thres = 0.15;      //13
const float normal_clothes_thres = 0.5; //14
const float face_thres = 0.5;           //15
const float play_phone_thres = 0.15;    //16
const float other_thres = 0.5;         //17
const float knife_thres = 0.15;         //18



/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
    printf("start test... \n");

    std::cout << "Number of arguments: " << argc << std::endl;
    
    std::unordered_set<int> filtered_classes; //存放要显示的类别号
    if (argc > 2 && std::strcmp(argv[1], "--classes") == 0){
        for(int i =0; i < argc - 2; i++){
            filtered_classes.insert(std::atoi(argv[2 + i]));
        }
    }else if (argc > 2 && std::strcmp(argv[1], "--classes") != 0){
        std::cout << "the argument must is --classes" << endl;
        return 0;
    }
     
    int ret = 0;
    Detect MultiDetecter;
    // 每个类别的筛选阈值
    std::vector<float> conf_thresholds {person_thres, head_helmet_thres, head_thres, reflective_clothes_thres, smoking_thres,
    calling_thres, falling_thres, face_mask_thres, car_thres, bicycle_thres, motorcycle_thres, fumes_thres, fire_thres, head_hat_thres, normal_clothes_thres,face_thres,play_phone_thres,other_thres,
    knife_thres};

    const char* models_path = "../models"; //所有模型所在的目录
    std::string strMultiDetect = "/yolov8s-p2-19classes-250epoch.rknn";
    std::string Multidetecter_model_path = models_path + strMultiDetect;
    ret = MultiDetecter.init_model(Multidetecter_model_path.c_str(), NMS_THRESHOLD, CONF_THRESHOLD, -1, conf_thresholds);
    
    cv::VideoCapture cap("rtsp://172.16.40.84:553/live", cv::CAP_FFMPEG);
    // cv::VideoCapture cap(11, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        printf("open video failed");
        return -1;
    }

    int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int video_fps = cap.get(cv::CAP_PROP_FPS);
    int video_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    printf("video info: width=%d, height=%d, fps=%d, frames=%d\n", video_width, video_height, video_fps, video_frames);

    BYTETracker tracker(video_fps, 50);
    while (true)
    {   
        cv::Mat orig_img;
        cap >> orig_img;

        if (orig_img.empty()) {
            continue;  // 跳过空帧
        }

        std::vector<int> ids;
        std::vector<float> conf;
        std::vector<cv::Rect> boxes;
        
        //预处理
        cv::resize(orig_img, orig_img, cv::Size(640, 360), 0, 0, cv::INTER_AREA);
        cv::Mat rgb_img;
        cv::cvtColor(orig_img, rgb_img, cv::COLOR_BGR2RGB);

        
        //
        TIMER process_time;
        process_time.tik();  //设置开始时间
        ret = MultiDetecter.detect(rgb_img, &ids, &conf, &boxes);

        /*为每个类别分别维护跟踪器,此处单独为0类进行跟踪*/
        vector<Object> objects;
        for(int i = 0; i < ids.size(); i++){
            if(ids[i] == 0){
                Object obj;
                obj.label = ids[i];
                obj.prob = conf[i];
                obj.rect = boxes[i];
                objects.push_back(obj);
            }
            else{
                continue;;
            }
        }
        vector<STrack> output_stracks = tracker.update(objects);
        

        process_time.tok();  //设置结束时间
        process_time.print_time("infer ");


        /*
        绘制检测框
        */
        // for (int i = 0; i < ids.size(); i++)
        // {   
        //     if(argc > 2 && filtered_classes.size() != 0){
        //         //显示指定的类别预测框
        //         if(filtered_classes.find(ids[i]) != filtered_classes.end()){
        //             cv::rectangle(orig_img, boxes[i], colors[ids[i]], 2, 1, 0);
        //             std::string conf_str = std::to_string(conf[i]);
        //             conf_str = conf_str.substr(0, conf_str.find('.') + 3); 
        //             std::string label = names[ids[i]] + ": " + conf_str;
        //             cv::putText(orig_img, label, cv::Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[ids[i]], 1);
        //         }
        //     }else{
        //         //如无指定，显示所有预测框
        //             cv::rectangle(orig_img, boxes[i], colors[ids[i]], 2, 1, 0);
        //             std::string conf_str = std::to_string(conf[i]);
        //             conf_str = conf_str.substr(0, conf_str.find('.') + 3);  // 保留conf小数后两位
        //             std::string label = names[ids[i]] + ": " + conf_str;
        //             cv::putText(orig_img, label, cv::Point(boxes[i].x, boxes[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[ids[i]], 1);
        //     }
        // }
        



        /*
        绘制跟踪框
        */
        for(int i = 0; i < output_stracks.size(); i++){
            if(argc > 2 && filtered_classes.size() != 0){
                if(filtered_classes.find(output_stracks[i].label) != filtered_classes.end()){
                  vector<float> tlwh = output_stracks[i].tlwh;
                  std::string conf_str = std::to_string(output_stracks[i].score);
                  conf_str = conf_str.substr(0, conf_str.find('.') + 3); 
                  std::string label = names[output_stracks[i].label] + ": " + conf_str;
                  cv::putText(orig_img, label, cv::Point(tlwh[0], tlwh[1] - 5),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
                  cv::putText(orig_img, "id: " + format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 15),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
                  cv::rectangle(orig_img,Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), colors[output_stracks[i].label], 2);
                }
            }else{
                vector<float> tlwh = output_stracks[i].tlwh;
                std::string conf_str = std::to_string(output_stracks[i].score);
                conf_str = conf_str.substr(0, conf_str.find('.') + 3); 
                std::string label = names[output_stracks[i].label] + ": " + conf_str;
                cv::putText(orig_img, label, cv::Point(tlwh[0], tlwh[1] - 5),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
                cv::putText(orig_img, "id: " + format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 15),cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[output_stracks[i].label], 1);
                cv::rectangle(orig_img,Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), colors[output_stracks[i].label], 2);                
            }
        }




        // Show result
        cv::imshow("detect", orig_img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    printf("end test... \n");

    return 0;
}

