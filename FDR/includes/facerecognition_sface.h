#ifndef FACE_RECOGNITION_SFACE_H
#define FACE_RECOGNITION_SFACE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "framecapture.h"
#include "facedetection_yunet.h"
#include <thread>
#include <pthread.h>
#include <mutex>

class FaceRecognitionSFace
{
public:
    FaceRecognitionSFace(const std::string &model_path, FrameCapture& frame_capture, FaceDetectionYunet& face_detector);
    ~FaceRecognitionSFace();
    void facerecognition(bool startrecognition = true);
    void release();
    cv::Mat getFrame();
    cv::Mat getTargetImage();
    cv::Mat getTargetFace();
private:
    cv::Ptr<cv::FaceRecognizerSF> model;
    std::string model_location;
    FrameCapture& frame_capture; // Reference to shared frame capture
    FaceDetectionYunet& face_detector; // Reference to face detector
    std::mutex frame_mutex; // Mutex for thread-safe frame access
    cv::Size input_size = cv::Size(320, 320); // Input size for the model
    float conf_threshold = 0.9f;
    float nms_threshold = 0.3f;
    int top_k = 5000;
    int backend_id = 0; // 0 for OpenCV, 1 for OpenVINO, 2 for CUDA
    int target_id = 0; // 0 for CPU, 1 for GPU
    int distance_type = 0; // 0 for L2, 1 for Cosine
    double threshold_cosine = 0.363;  // Increased for stricter matching
    double threshold_norml2 = 1.128;  // Decreased for stricter matching
    cv::Mat infer_recog_frames;
    cv::Mat resultant_recognition_frame;
    cv::Mat recog_frame;
    cv::Mat target_image;
    cv::Mat target_face;
    cv::Mat draw_frame_box(const cv::Mat& image, const cv::Mat& result, float fps = -1.0f, bool is_recognition = false);
    int x;
    int y;
    int w;
    int h;

};


#endif // FACE_RECOGNITION_SFACE_H