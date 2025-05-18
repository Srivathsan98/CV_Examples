#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include "facedetection_yunet.h"
#include "facerecognition_sface.h"
#include "framecapture.h"

int main() {
    std::string face_detection_model = "./models/face_detection_model.onnx";
    std::string face_recognition_model = "./models/face_recognition_model.onnx";

    try {
        // Initialize frame capture
        FrameCapture frame_capture;
        frame_capture.start();

        // Initialize the face detection model with frame capture
        FaceDetectionYunet face_detection(face_detection_model, frame_capture);
        
        // Initialize face recognition with frame capture and face detection
        FaceRecognitionSFace face_recognition(face_recognition_model, frame_capture, face_detection);

        char key;
        bool detection_active = false;
        bool recognition_active = false;
        std::thread detection_thread;
        std::thread recognition_thread;

        while (true) {
            std::cout << "Press 'd' for detection, 'r' for recognition, 'q' to quit: ";
            std::cin >> key;

            if (key == 'd') {
                if (!detection_active) {
                    // Start face detection in a new thread
                    detection_thread = std::thread([&face_detection]() {
                        face_detection.detectFaces(true);
                    });
                    pthread_setname_np(detection_thread.native_handle(), "FaceDetectionThread");
                    detection_active = true;
                    std::cout << "Detection started." << std::endl;
                } else {
                    std::cout << "Detection is already running." << std::endl;
                }
            } else if (key == 'r') {
                if (detection_active) {
                    if (!recognition_active) {
                        // Start face recognition in a new thread
                        recognition_thread = std::thread([&face_recognition]() {
                            face_recognition.facerecognition(true);
                        });
                        pthread_setname_np(recognition_thread.native_handle(), "FaceRecognitionThread");
                        recognition_active = true;
                        std::cout << "Recognition started." << std::endl;
                    } else {
                        std::cout << "Recognition is already running." << std::endl;
                    }
                } else {
                    std::cout << "Please start detection first." << std::endl;
                }
            } else if (key == 'q') {
                if (recognition_active && recognition_thread.joinable()) {
                    recognition_thread.join();
                }
                if (detection_active && detection_thread.joinable()) {
                    detection_thread.join();
                }
                face_recognition.release();
                face_detection.release();
                frame_capture.stop();
                break;
            } else {
                std::cout << "Invalid input. Please try again." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}