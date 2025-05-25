// File: facerecognizer.h
#ifndef FACERECOGNIZER_H
#define FACERECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

class FaceRecognizer {
public:
    FaceRecognizer(const std::string& modelPath);
    std::string recognize(const cv::Mat& frame, const cv::Rect& face);

private:
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
    cv::Mat target_embedding;
};

#endif // FACERECOGNIZER_H