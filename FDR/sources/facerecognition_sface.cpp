#include "facerecognition_sface.h"

FaceRecognitionSFace::FaceRecognitionSFace(const std::string &model_path, FrameCapture& frame_capture, FaceDetectionYunet& face_detector)
    : frame_capture(frame_capture), face_detector(face_detector)
{
    model_location = model_path;
    // Load the model
    model = cv::FaceRecognizerSF::create(model_path, "", backend_id, target_id);
    if (model.empty()) {
        throw std::runtime_error("Could not load the face recognition model");
    }
    std::cout << "Face recognition model loaded successfully." << std::endl;
}
void FaceRecognitionSFace::release()
{
    if (!model.empty()) {
        model.release();
        std::cout << "Face recognition model released." << std::endl;
    }
}

FaceRecognitionSFace::~FaceRecognitionSFace()
{
    release();
}

void FaceRecognitionSFace::facerecognition(bool startrecognition)
{
    if (model.empty()) {
        throw std::runtime_error("Face recognition model is not loaded");
    }

    // Get the target image and face
    target_image = cv::imread("models/target_image.jpg"); // Load the target image
    if (target_image.empty()) {
        std::cerr << "Error: Could not load target image" << std::endl;
        return;
    }

    // Detect face in target image
    cv::Mat target_faces;
    face_detector.detect(target_image, target_faces);
    if (target_faces.empty()) {
        std::cerr << "Error: No face detected in target image" << std::endl;
        return;
    }

    // Extract face alignment points
    cv::Mat aligned_face;
    model->alignCrop(target_image, target_faces.row(0), aligned_face);

    // Extract features from aligned face
    cv::Mat target_features;
    model->feature(aligned_face, target_features);
    std::cout << "Target face features extracted successfully." << std::endl;

    while (startrecognition) {
        // Get the latest frame with detected faces from face detector
        cv::Mat frame = frame_capture.getFrame();
        if (frame.empty()) {
            std::cerr << "Warning: Empty frame received" << std::endl;
            continue;
        }

        // Detect faces in current frame
        cv::Mat faces;
        face_detector.detect(frame, faces);

        cv::Mat output = frame.clone();
        if (!faces.empty()) {
            // Process each detected face
            for (int i = 0; i < faces.rows; i++) {
                // Get face alignment points
                cv::Mat aligned_face;
                model->alignCrop(frame, faces.row(i), aligned_face);

                // Extract features
                cv::Mat features;
                model->feature(aligned_face, features);

                // Match with target face
                float cosine_score = model->match(target_features, features, cv::FaceRecognizerSF::DisType::FR_COSINE);
                float l2_score = model->match(target_features, features, cv::FaceRecognizerSF::DisType::FR_NORM_L2);

                // Check if face matches
                bool is_same_person = (cosine_score >= threshold_cosine) || (l2_score <= threshold_norml2);

                // Draw bounding box and recognition result
                int x = static_cast<int>(faces.at<float>(i, 0));
                int y = static_cast<int>(faces.at<float>(i, 1));
                int w = static_cast<int>(faces.at<float>(i, 2));
                int h = static_cast<int>(faces.at<float>(i, 3));

                cv::Scalar color = is_same_person ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(output, cv::Rect(x, y, w, h), color, 2);

                std::string label = cv::format("Cosine: %.2f, L2: %.2f", cosine_score, l2_score);
                cv::putText(output, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                
                std::string match_text = is_same_person ? "MATCH" : "NO MATCH";
                cv::putText(output, match_text, cv::Point(x, y + h + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
        }

        // Display the frame with recognition results
        cv::imshow("Face Recognition", output);
        char key = (char)cv::waitKey(1);
        if (key >= 0) {
            break; // Exit the loop if a key is pressed
        }
    }
}
cv::Mat FaceRecognitionSFace::draw_frame_box(const cv::Mat& image, const cv::Mat& result, float fps)
{
    cv::Mat image_copy = image.clone();
    for (int i = 0; i < result.rows; ++i) {
        int x1 = static_cast<int>(result.at<float>(i, 0));
        int y1 = static_cast<int>(result.at<float>(i, 1));
        int w = static_cast<int>(result.at<float>(i, 2));
        int h = static_cast<int>(result.at<float>(i, 3));
        float conf = result.at<float>(i, 4);
        cv::rectangle(image_copy, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("Conf: %.2f", conf);
        cv::putText(image_copy, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    if (fps > 0) {
        std::string fps_text = cv::format("FPS: %.2f", fps);
        cv::putText(image_copy, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    return image_copy;
}
cv::Mat FaceRecognitionSFace::getFrame()
{
    std::lock_guard<std::mutex> lock(frame_mutex);
    return resultant_recognition_frame.clone();
}
cv::Mat FaceRecognitionSFace::getTargetImage()
{
    std::lock_guard<std::mutex> lock(frame_mutex);
    return target_image.clone();
}
cv::Mat FaceRecognitionSFace::getTargetFace()
{
    std::lock_guard<std::mutex> lock(frame_mutex);
    return target_face.clone();
}