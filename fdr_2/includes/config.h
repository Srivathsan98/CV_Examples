#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <string>

//configs
std::string facedetection_modelpath = "../models/face_detection_model.onnx";
std::string facerecognition_modelpath = "../models/face_recognition_model.onnx";
std::string target_image_path = "../models/target_image.jpg";
std::string target_image_name = "target.jpg";
#endif