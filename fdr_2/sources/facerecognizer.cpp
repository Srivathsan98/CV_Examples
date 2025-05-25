// // File: facerecognizer.cpp
// #include "facerecognizer.h"

// FaceRecognizer::FaceRecognizer(const std::string& modelPath) {
//     recognizer = cv::FaceRecognizerSF::create(modelPath, "");

//     // Load target image
//     cv::Mat target = cv::imread("target.jpg");
//     std::vector<cv::Rect> face;
//     cv::CascadeClassifier cascade("haarcascade_frontalface_default.xml");
//     cascade.detectMultiScale(target, face);

//     if (!face.empty()) {
//         cv::Mat aligned;
//         recognizer->alignCrop(target, face[0], aligned);
//         recognizer->feature(aligned, targetEmbedding);
//     }
// }

// std::string FaceRecognizer::recognize(const cv::Mat& frame, const cv::Rect& face) {
//     if (targetEmbedding.empty()) return "Unknown";
//     cv::Mat aligned, feat;
//     recognizer->alignCrop(frame, face, aligned);
//     recognizer->feature(aligned, feat);
//     double score = recognizer->match(feat, targetEmbedding);
//     return score > 0.5 ? "Target" : "Unknown";
// }

#include "facerecognizer.h"

FaceRecognizer::FaceRecognizer(const std::string& model_path) {
    recognizer = cv::FaceRecognizerSF::create(model_path, "");
    
    // Load target image
    cv::Mat target = cv::imread("/home/pvsp/OpenCV_GST_Practice/CV/fdr_2/models/target_image.jpg");
    if (target.empty()) {
        std::cerr << "Failed to load target image!" << std::endl;
        exit(1);
    }

    // Detect face in target image using a face detector (YuNet or similar)
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("../models/face_detection_model.onnx", "", cv::Size(320, 320), 0.9f, 0.3f, 5000);
    detector->setInputSize(target.size());
    // detector.setTopK(1);

    cv::Mat faces;
    detector->detect(target, faces);

    if (faces.rows == 0) {
        std::cerr << "No face found in target image!" << std::endl;
        exit(1);
    }

    cv::Rect faceRect(
        static_cast<int>(faces.at<float>(0, 0)),
        static_cast<int>(faces.at<float>(0, 1)),
        static_cast<int>(faces.at<float>(0, 2)),
        static_cast<int>(faces.at<float>(0, 3))
    );

    // Wrap the faceRect as InputArray (by converting it to a Mat)
    cv::Mat faceMat(1, 4, CV_32FC1);
    faceMat.at<float>(0, 0) = static_cast<float>(faceRect.x);
    faceMat.at<float>(0, 1) = static_cast<float>(faceRect.y);
    faceMat.at<float>(0, 2) = static_cast<float>(faceRect.width);
    faceMat.at<float>(0, 3) = static_cast<float>(faceRect.height);

    cv::Mat aligned;
    recognizer->alignCrop(target, faceMat, aligned);
    recognizer->feature(aligned, target_embedding);
}

std::string FaceRecognizer::recognize(const cv::Mat& frame, const cv::Rect& face) {
    // Wrap the face rect for alignCrop
    cv::Mat faceMat(1, 4, CV_32FC1);
    faceMat.at<float>(0, 0) = static_cast<float>(face.x);
    faceMat.at<float>(0, 1) = static_cast<float>(face.y);
    faceMat.at<float>(0, 2) = static_cast<float>(face.width);
    faceMat.at<float>(0, 3) = static_cast<float>(face.height);

    cv::Mat aligned;
    recognizer->alignCrop(frame, faceMat, aligned);

    cv::Mat embedding;
    recognizer->feature(aligned, embedding);

    double score = recognizer->match(embedding, target_embedding);
    return score > 0.3 ? "Match" : "Unknown";  // Threshold can be tuned
}
