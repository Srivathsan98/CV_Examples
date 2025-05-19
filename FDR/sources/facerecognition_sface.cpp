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
    if (model.empty())
    {
        throw std::runtime_error("Face recognition model is not loaded");
    }

    // Get the target image and face
    target_image = cv::imread("models/target_image.jpg"); // Load the target image
    if (target_image.empty())
    {
        std::cerr << "Error: Could not load target image" << std::endl;
        return;
    }

    // Detect face in target image
    cv::Mat target_faces;
    face_detector.detect(target_image, target_faces);
    if (target_faces.empty())
    {
        std::cerr << "Error: No face detected in target image" << std::endl;
        return;
    }

    // Extract face alignment points
    cv::Mat target_aligned_face;
    model->alignCrop(target_image, target_faces.row(0), target_aligned_face);

    // Extract features from aligned face
    cv::Mat target_features;
    model->feature(target_aligned_face, target_features);
    std::cout << "Target face features extracted successfully." << std::endl;

    cv::normalize(target_features, target_features);

    while (startrecognition)
    {
        // Get the latest frame with detected faces from face detector
        cv::Mat frame = frame_capture.getFrame();
        if (frame.empty())
        {
            std::cerr << "Warning: Empty frame received" << std::endl;
            continue;
        }

        // Detect faces in current frame
        cv::Mat faces;
        face_detector.detect(frame, faces);

        cv::Mat output = frame.clone();
        if (!faces.empty())
        {
            // Process each detected face
            for (int i = 0; i < faces.rows; i++)
            {

                // Draw bounding box and recognition result
                x = static_cast<int>(faces.at<float>(i, 0));
                y = static_cast<int>(faces.at<float>(i, 1));
                w = static_cast<int>(faces.at<float>(i, 2));
                h = static_cast<int>(faces.at<float>(i, 3));
                float conf = faces.at<float>(i, 14);
                std::cout << cv::format("%d: x=%d, y=%d, w=%d, h=%d, conf=%.4f", i, x, y, w, h, conf) << std::endl;

            // Get face alignment points
            cv::Mat aligned_face;
            model->alignCrop(frame, faces.row(i), aligned_face);
            if (aligned_face.empty()) {
    std::cerr << "Aligned target face is empty" << std::endl;
    return;
}

            // Extract features
            cv::Mat features;
            model->feature(aligned_face, features);

            cv::normalize(features, features);

            // Match with target face
            float cosine_score = model->match(target_features, features, cv::FaceRecognizerSF::DisType::FR_COSINE);
            float l2_score = model->match(target_features, features, cv::FaceRecognizerSF::DisType::FR_NORM_L2);

            // Check if face matches
            bool is_same_person = (cosine_score >= threshold_cosine) && (l2_score <= threshold_norml2);  // Using AND for stricter matching

            // Draw bounding box and match result
            cv::Scalar color = is_same_person ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(output, cv::Rect(x, y, w, h), color, 2);

            std::string label = cv::format("Cosine: %.2f, L2: %.2f", cosine_score, l2_score);
            cv::putText(output, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

            std::string match_text = is_same_person ? "MATCH" : "NO MATCH";
            cv::putText(output, match_text, cv::Point(x, y + h + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

            std::cout << cv::format("Face %d: Cosine = %.4f, L2 = %.4f --> Match: %s\n",
    i, cosine_score, l2_score, is_same_person ? "YES" : "NO") << std::endl;
            // auto target_face_image = draw_frame_box(target_image, target_faces.row(0), -1.0f);
            // auto recog_face_image = draw_frame_box(frame, faces, -1.0f, is_same_person);
            // // cv::hconcat(target_face_image, recog_face_image, output);
            // cv::putText(output, is_same_person ? "MATCH" : "NO MATCH", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            // cv::putText(output, cv::format("Cosine: %.2f, L2: %.2f", cosine_score, l2_score), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            // std::cout << "Cosine: " << cosine_score << ", L2: " << l2_score << std::endl;
            }


        }

        // Display the frame with recognition results
        cv::imshow("Face Recognition", output);
        char key = (char)cv::waitKey(1);
        if (key >= 0)
        {
            break; // Exit the loop if a key is pressed
        }
    }
}
cv::Mat FaceRecognitionSFace::draw_frame_box(const cv::Mat& image, const cv::Mat& result, float fps, bool is_recognition)
{
    cv::Mat image_copy = image.clone();
    static const cv::Scalar matched_box_color{0, 255, 0};
    static const cv::Scalar mismatched_box_color{0, 0, 255};
    if (fps > 0) {
    for (int i = 0; i < result.rows; ++i) {
        int x1 = static_cast<int>(result.at<float>(i, 0));
        int y1 = static_cast<int>(result.at<float>(i, 1));
        int w = static_cast<int>(result.at<float>(i, 2));
        int h = static_cast<int>(result.at<float>(i, 3));
        float conf = result.at<float>(i, 4);
        cv::Scalar box_color = is_recognition ? matched_box_color : mismatched_box_color;
        cv::rectangle(image_copy, cv::Rect(x1, y1, w, h), box_color, 2);
        std::string label = cv::format("Conf: %.2f", conf);
        cv::putText(image_copy, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2);
    
        std::string fps_text = cv::format("FPS: %.2f", fps);
        cv::putText(image_copy, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
    }
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