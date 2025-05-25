// File: main.cpp
#include "framecapture.h"
#include "facedetector.h"
#include "facerecognizer.h"

#include <thread>
#include <atomic>

std::atomic<char> command('\0');

void inputThread() {
    while (true) {
        char c = std::cin.get();
        if (c != '\n') {
            command.store(c);
        }
        if (c == 'q') {
            false;
        }
    }
}

int main() {
    std::thread t(inputThread);
    FrameCapture frameCapture;
    FaceDetector faceDetector("/home/pvsp/OpenCV_GST_Practice/CV/fdr_2/models/face_detection_model.onnx");
    FaceRecognizer faceRecognizer("/home/pvsp/OpenCV_GST_Practice/CV/fdr_2/models/face_recognition_model.onnx");
    // const std::string targetImagePath = "models/target.jpg";

    std::atomic<bool> running{true};
    std::thread captureThread(&FrameCapture::captureLoop, &frameCapture, std::ref(running));

    // if(targetImagePath.empty()) {
    //     std::cerr << "Error: Target image path is empty." << std::endl;
    //     return -1;
    // }
    // cv::Mat targetImage = cv::imread(targetImagePath);
    // if (targetImage.empty()) {
    //     std::cerr << "Error: Could not load target image from " << targetImagePath << std::endl;
    //     return -1;
    // }

    // faceDetector.setFrameInputSize(targetImage.size());
    // faceDetector.setdetectionTopK(1);
    // cv::Mat target_face = faceDetector.infer(targetImage);

    // std::cout << "Press 'd' for detection, 'r' for recognition, 'q' to quit.\n";
    // cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
    // cv::namedWindow("TestWindow");    

    // while (running.load()) {
    //     // std::cout << "Waiting for input...\n";
    //     // cv::imshow("TestWindow", img);
    //     // int key = cv::waitKey(10);
    //     char key = command.load();
    //     if (key != -1) {
    //     std::cout << "Key pressed: " << key << " (char: '" << (char)key << "')\n";
    // }
    //     if (key == 'd') {
    //         std::cout << "Detecting faces...\n";
    //         cv::Mat frame = frameCapture.getLatestFrame();
    //         std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
    //         frameCapture.visualize(frame, faces, {});
    //         command.store('\0'); // Reset command after processing
    //     } else if (key == 'r') {
    //         cv::Mat frame = frameCapture.getLatestFrame();
    //         std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
    //         std::vector<std::string> names;
    //         for (const auto& face : faces) {
    //             names.push_back(faceRecognizer.recognize(frame, face));
    //         }
    //         frameCapture.visualize(frame, faces, names);
    //         command.store('\0'); // Reset command after processing
    //     } else if (key == 'q') {
    //         running = false;
    //     }
    //     cv::waitKey(10); // Allow OpenCV to process events
    // }
    // t.join();

    while (cv::waitKey(1) < 0) {
        cv::Mat frame = frameCapture.getLatestFrame();
        if (frame.empty()) continue;

        std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
        std::vector<std::string> names;

        for (const auto& face : faces) {
            names.push_back(faceRecognizer.recognize(frame, face));
        }

        frameCapture.visualize(frame, faces, names);
    }
    running = false; // Stop the capture loop
    if (t.joinable()) {
        t.join();
    }
    if (captureThread.joinable())  // Ensure the capture thread is joined
    {
        /* code */
        captureThread.join();
    }
    
    
    return 0;
}
