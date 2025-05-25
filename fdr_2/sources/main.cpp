// File: main.cpp
#include "framecapture.h"
#include "facedetector.h"
#include "facerecognizer.h"
#include "config.h"
#include <thread>
#include <atomic>
/************************************************************************ */
// std::atomic<char> command('\0');

// void inputThread() {
//     while (true) {
//         char c = std::cin.get();
//         if (c != '\n') {
//             command.store(c);
//         }
//         if (c == 'q') {
//             false;
//         }
//     }
// }

// int main() {
//     std::thread t(inputThread);
//     FrameCapture frameCapture;
//     FaceDetector faceDetector("/home/pvsp/OpenCV_GST_Practice/CV/fdr_2/models/face_detection_model.onnx");
//     FaceRecognizer faceRecognizer("/home/pvsp/OpenCV_GST_Practice/CV/fdr_2/models/face_recognition_model.onnx");
//     // const std::string targetImagePath = "models/target.jpg";

//     std::atomic<bool> running{true};
//     std::thread captureThread(&FrameCapture::captureLoop, &frameCapture, std::ref(running));

//     // std::cout << "Press 'd' for detection, 'r' for recognition, 'q' to quit.\n";

//     // while (running.load()) {
//     //     // std::cout << "Waiting for input...\n";
//     //     // cv::imshow("TestWindow", img);
//     //     // int key = cv::waitKey(10);
//     //     char key = command.load();
//     //     if (key != -1) {
//     //     std::cout << "Key pressed: " << key << " (char: '" << (char)key << "')\n";
//     // }
//     //     if (key == 'd') {
//     //         std::cout << "Detecting faces...\n";
//     //         cv::Mat frame = frameCapture.getLatestFrame();
//     //         std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
//     //         frameCapture.visualize(frame, faces, {});
//     //         command.store('\0'); // Reset command after processing
//     //     } else if (key == 'r') {
//     //         cv::Mat frame = frameCapture.getLatestFrame();
//     //         std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
//     //         std::vector<std::string> names;
//     //         for (const auto& face : faces) {
//     //             names.push_back(faceRecognizer.recognize(frame, face));
//     //         }
//     //         frameCapture.visualize(frame, faces, names);
//     //         command.store('\0'); // Reset command after processing
//     //     } else if (key == 'q') {
//     //         running = false;
//     //     }
//     //     cv::waitKey(10); // Allow OpenCV to process events
//     // }
//     // t.join();

//     while (cv::waitKey(1) < 0) {
//         cv::Mat frame = frameCapture.getLatestFrame();
//         if (frame.empty()) continue;

//         std::vector<cv::Rect> faces = faceDetector.detectFaces(frame);
//         std::vector<std::string> names;

//         for (const auto& face : faces) {
//             names.push_back(faceRecognizer.recognize(frame, face));
//         }

//         frameCapture.visualize(frame, faces, names);
//     }
//     running = false; // Stop the capture loop
//     if (t.joinable()) {
//         t.join();
//     }
//     if (captureThread.joinable())  // Ensure the capture thread is joined
//     {
//         /* code */
//         captureThread.join();
//     }
    
    
//     return 0;
// }
/***************************************************************************** */

int main()
{
    std::atomic<bool> running(true);

    FrameCapture m_frameCapture;
    // FaceDetector m_faceDetector;
    // FaceRecognizer m_faceRecognizer;

    std::thread captureThread(&FrameCapture::captureLoop, &m_frameCapture, std::ref(running));
    // Set thread name (max 15 chars on Linux)
    pthread_setname_np(captureThread.native_handle(), "CaptureLoop");

    auto facedetector = FaceDetector(facedetection_modelpath, cv::Size(320, 320), 0.9f, 0.3f, 100, 0, 0);
    std::cout << "Face detection loaded" << std::endl;
    auto face_recognizer = FaceRecognizer(facerecognition_modelpath, 0, 0, 0);
    std::cout << "Face recognition loaded" << std::endl;

    if(target_image_path.empty())
    {
        std::cerr << "Target image path is empty. Please set the target image path in the config.h file." << std::endl;
        return -1;
    }
    cv::Mat target_image = cv::imread(target_image_path);
    if (target_image.empty())
    {
        std::cerr << "Failed to load target image from: " << target_image_path << std::endl;
        return -1;
    }

    facedetector.setFrameInputSize(target_image.size());
    facedetector.setdetectionTopK(3);
    cv::Mat target_face = facedetector.infer(target_image);

    cv::Mat target_features = face_recognizer.extractfeatures(target_image, target_face);

    const int w = m_frameCapture.getCameraSize().first;
    const int h = m_frameCapture.getCameraSize().second;
    facedetector.setFrameInputSize(cv::Size(w, h));

    // std::cout << cv::getBuildInformation() << std::endl;

    cv::VideoWriter gst_writer;
    // std::string gst_pipeline = 
    // "appsrc caps=video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! "
    // "videoconvert ! jpegenc ! rtpjpegpay ! "
    // "udpsink host=127.0.0.1 port=5000";


    // gst_writer.open(gst_pipeline, 0, 30.0, cv::Size(w * 2, h), true); // w*2 because of hconcat
    // if (!gst_writer.isOpened())
    // {
    //     std::cerr << "Failed to open GStreamer pipeline for video output." << std::endl;
    //     return -1;
    // }

    cv::Mat query_frame;
    static bool gst_writer_initialized = false;

    while(cv::waitKey(1) < 0 && running.load())
    {
        query_frame = m_frameCapture.getLatestFrame();
        if (query_frame.empty()) 
        {
            std::cerr << "Empty frame captured, skipping..." << std::endl;
            continue;
        }
        cv::Mat query_faces = facedetector.infer(query_frame);
        if (query_faces.empty())
        {
            // std::cerr << "No faces detected in the query frame." << std::endl;
            continue;
        }

        for(int i = 0; i < query_faces.rows; ++i)
        {
            int x1 = static_cast<int>(query_faces.at<float>(i, 0));
            int y1 = static_cast<int>(query_faces.at<float>(i, 1));
            int w = static_cast<int>(query_faces.at<float>(i, 2));
            int h = static_cast<int>(query_faces.at<float>(i, 3));
            float conf = query_faces.at<float>(i, 14);
            // std::cout << cv::format("Face %d: x=%d, y=%d, w=%d, h=%d, conf=%.2f", i, x1, y1, w, h, conf) << std::endl;
        }

        std::vector<std::pair<double, bool>> matches;


        cv::Mat query_featrues = face_recognizer.extractfeatures(query_frame, query_faces.row(0));
        const auto match = face_recognizer.matchFeatures(target_features, query_featrues);

        auto vis_target = m_frameCapture.visualize(target_image, target_face, {{1.0, true}}, -0.1f, cv::Size(w, h));
        auto vis_query = m_frameCapture.visualize(query_frame, query_faces, {match}, 30.00);
        cv::Mat output_image;
        cv::hconcat(vis_target, vis_query, output_image);
        // cv::imshow("Face Recognition", vis_query);
        // gst_writer.write(vis_query);
        
    if (!gst_writer_initialized)
    {
        int width = output_image.cols;
        int height = output_image.rows;
        // std::ostringstream pipeline;
        // pipeline << "appsrc is-live=true block=true caps=video/x-raw,width=" << width
        //          << ",height=" << height << ",framerate=30/1 ! "
        //          << "videoconvert ! jpegenc ! rtpjpegpay ! "
        //          << "udpsink host=127.0.0.1 port=5000";
        std::ostringstream pipeline;
pipeline << "appsrc is-live=true block=true format=3 caps=video/x-raw,format=BGR,width=" << width
         << ",height=" << height << ",framerate=30/1 ! "
         << "videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=500 ! "
         << "rtph264pay config-interval=1 pt=96 ! "
         << "udpsink host=127.0.0.1 port=5000";


        // gst_writer.open(pipeline.str(), 0, 30.0, cv::Size(width, height), true);
        gst_writer.open(pipeline.str(), cv::CAP_GSTREAMER, 30.0, cv::Size(width, height), true);

        if (!gst_writer.isOpened())
        {
            std::cerr << "Failed to open GStreamer pipeline for video output." << std::endl;
            break;
        }
        gst_writer_initialized = true;
    }

    if (output_image.channels() != 3)
    {
        cv::cvtColor(output_image, output_image, cv::COLOR_GRAY2BGR);
    }
    

    if(output_image.empty())
    {
        std::cerr << "Output image is empty, skipping write." << std::endl;
        continue;
    }
    else
    {
        gst_writer.write(output_image);
        std::cout << "Writing output image to GStreamer pipeline." << std::endl;
    }
    }
    running = false; // Stop the capture loop
    if (captureThread.joinable())
    {
        captureThread.join();
    }
    std::cout << "Exiting program..." << std::endl;
    cv::destroyAllWindows();
    return 0;
}