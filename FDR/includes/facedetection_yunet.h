#ifndef FACE_DETECTION_YUNET_H
#define FACE_DETECTION_YUNET_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "framecapture.h"

class FaceDetectionYunet {
public:
    FaceDetectionYunet(const std::string &model_path, FrameCapture& frame_capture);
    ~FaceDetectionYunet();
    
    void detectFaces(bool startdetection = true);
    void release();
    cv::Mat getFrame();
    void detect(const cv::Mat& frame, cv::Mat& faces); // Public method to detect faces in a frame

private:
    std::string model_location;
    FrameCapture& frame_capture; // Reference to shared frame capture
    std::mutex frame_mutex; // Mutex for thread-safe frame access

public: // Make model accessible to face recognition
    cv::Ptr<cv::FaceDetectorYN> model;
    
    cv::Size input_size = cv::Size(320, 320); // Input size for the model
    float conf_threshold = 0.6f;
    float nms_threshold = 0.3f;
    int top_k = 5000;
    int backend_id = 0; // 0 for OpenCV, 1 for OpenVINO, 2 for CUDA
    int target_id = 0; // 0 for CPU, 1 for GPU
    cv::Mat infer_detect_frames;
    cv::Mat resultant_detection_frame;
    cv::Mat detect_frame;
    cv::Mat draw_frame_box(const cv::Mat& image, const cv::Mat& result, float fps = -1.0f);
};

#endif // FACE_DETECTION_YUNET_H