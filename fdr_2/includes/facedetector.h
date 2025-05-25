// File: facedetector.h
#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

class FaceDetector {
public:
    FaceDetector(const std::string& modelPath);
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    void setFrameInputSize(const cv::Size &size);
    void setdetectionTopK(int topK);
    cv::Mat infer(const cv::Mat &image);

private:
    cv::Ptr<cv::FaceDetectorYN> detector;
};

#endif // FACEDETECTOR_H