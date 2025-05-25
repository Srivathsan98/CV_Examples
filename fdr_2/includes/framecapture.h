// File: framecapture.h
#ifndef FRAMECAPTURE_H
#define FRAMECAPTURE_H

#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
class FrameCapture {
public:
    FrameCapture();
    void captureLoop(std::atomic<bool>& running);
    cv::Mat getLatestFrame();
    void visualize(const cv::Mat& frame, const std::vector<cv::Rect>& faces, const std::vector<std::string>& names);

private:
    cv::VideoCapture cap;
    cv::Mat latestFrame;
    std::mutex mtx;
};

#endif // FRAMECAPTURE_H