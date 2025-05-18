#include "facedetection_yunet.h"
#include <opencv2/opencv.hpp>

FaceDetectionYunet::FaceDetectionYunet(const std::string &model_path, FrameCapture& frame_capture)
    : frame_capture(frame_capture) {
    model_location = model_path;
    // Load the model
    model = cv::FaceDetectorYN::create(model_path, "", input_size, conf_threshold, nms_threshold, top_k, backend_id, target_id);
    if (model.empty()) {
        throw std::runtime_error("Could not load the face detection model");
    }
    std::cout << "Face detection model loaded successfully." << std::endl;
}

FaceDetectionYunet::~FaceDetectionYunet() {
    release();
}

void FaceDetectionYunet::release() {
    if (!model.empty()) {
        model.release();
        std::cout << "Face detection model released." << std::endl;
    }
}
cv::Mat FaceDetectionYunet::getFrame() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    return resultant_detection_frame.clone();
}

void FaceDetectionYunet::detect(const cv::Mat& frame, cv::Mat& faces) {
    if (model.empty()) {
        throw std::runtime_error("Face detection model is not loaded");
    }
    
    // Resize frame to match model's input size
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, input_size);
    
    // Set input size and detect faces
    model->setInputSize(input_size);
    model->detect(resized_frame, faces);
    
    // Scale back the face coordinates to original frame size
    if (!faces.empty()) {
        float scale_x = static_cast<float>(frame.cols) / input_size.width;
        float scale_y = static_cast<float>(frame.rows) / input_size.height;
        
        for (int i = 0; i < faces.rows; i++) {
            faces.at<float>(i, 0) *= scale_x;
            faces.at<float>(i, 1) *= scale_y;
            faces.at<float>(i, 2) *= scale_x;
            faces.at<float>(i, 3) *= scale_y;
            
            // Scale landmarks if present
            for (int j = 4; j < faces.cols; j += 2) {
                faces.at<float>(i, j) *= scale_x;
                faces.at<float>(i, j + 1) *= scale_y;
            }
        }
    }
}

void FaceDetectionYunet::detectFaces(bool startdetection) {
    if (model.empty()) {
        throw std::runtime_error("Face detection model is not loaded");
    }

    while(startdetection) {
        detect_frame = frame_capture.getFrame(); // Get the latest frame
        if (detect_frame.empty()) {
            std::cerr << "Warning: Empty frame received" << std::endl;
            continue;
        }

        // Detect faces using the common detect method
        detect(detect_frame, infer_detect_frames);
        
        resultant_detection_frame = detect_frame.clone();
        if (!infer_detect_frames.empty()) {
            // Draw the detected faces on the frame
            resultant_detection_frame = draw_frame_box(detect_frame, infer_detect_frames);
        }

        // Display the frame with detected faces
        cv::imshow("Face Detection", resultant_detection_frame);
        char key = (char)cv::waitKey(1);
        if (key >= 0) {
            break; // Exit the loop if a key is pressed
        }
    }
}

cv::Mat FaceDetectionYunet::draw_frame_box(const cv::Mat& image, const cv::Mat& result, float fps) {
    cv::Mat output = image.clone();

    // Draw FPS on the image
    if (fps > 0) {
        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(output, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    }
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(  0,   0, 255), // left eye
        cv::Scalar(  0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(  0, 255, 255)  // left mouth corner
    };

    // Draw bounding boxes around detected faces
    for (int i = 0; i < result.rows; i++) {
        // The model returns coordinates directly in pixels
        int x1 = static_cast<int>(result.at<float>(i, 0));
        int y1 = static_cast<int>(result.at<float>(i, 1));
        int w = static_cast<int>(result.at<float>(i, 2));
        int h = static_cast<int>(result.at<float>(i, 3));
        float conf = result.at<float>(i, 14);

        // Draw rectangle and confidence
        cv::rectangle(output, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("Face %.2f", conf);
        cv::putText(output, label, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        // Draw landmarks if available (5 landmarks: left eye, right eye, nose, left mouth, right mouth)
        for (int j = 0; j < 5; j++) {
            int landmark_x = static_cast<int>(result.at<float>(i, 4 + j * 2));
            int landmark_y = static_cast<int>(result.at<float>(i, 4 + j * 2 + 1));
            cv::circle(output, cv::Point(landmark_x, landmark_y), 2, landmark_color[j], -1);
        }
    }
    return output;
}