#ifndef FRAME_CAPTURE_H
#define FRAME_CAPTURE_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <thread>
#include <pthread.h>
#include <mutex>

class FrameCapture {
private:
    cv::VideoCapture cap; // OpenCV VideoCapture object
    std::thread capture_thread; // Thread for capturing frames
    bool runthread = false; // Flag to control the capture thread
    cv::Mat current_frame; // Current frame buffer
    std::mutex frame_mutex; // Mutex for thread-safe frame access

public:
    FrameCapture(); // Constructor
    ~FrameCapture(); // Destructor
    
    void start(); // Start capturing frames
    void stop(); // Stop capturing frames
    cv::Mat getFrame(); // Get the latest frame (thread-safe)
    bool isRunning() const { return runthread; }

private:
    void captureLoop(); // Frame capture loop
};

#endif // FRAME_CAPTURE_H
