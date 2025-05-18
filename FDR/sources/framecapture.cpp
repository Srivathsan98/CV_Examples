#include "framecapture.h"

FrameCapture::FrameCapture() {}

FrameCapture::~FrameCapture() {
    stop();
}

void FrameCapture::start() {
    // Open the default camera
    cap.open(0);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open camera");
    }
    
    runthread = true;
    capture_thread = std::thread(&FrameCapture::captureLoop, this);
    pthread_setname_np(capture_thread.native_handle(), "FrameCaptureThread");
}

void FrameCapture::stop() {
    runthread = false;
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (cap.isOpened()) {
        cap.release();
    }
    cv::destroyWindow("Camera Feed");
}

cv::Mat FrameCapture::getFrame() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    return current_frame.clone();
}

void FrameCapture::captureLoop() {
    while (runthread) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            continue;
        }

        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured" << std::endl;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            current_frame = frame.clone();
        }

        cv::imshow("Camera Feed", frame);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
}