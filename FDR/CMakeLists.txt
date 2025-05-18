cmake_minimum_required(VERSION 3.10)
project(FaceDetectionRecognition)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS}
                  ${CMAKE_SOURCE_DIR}/includes)

# Add source files
set(SOURCES
    sources/main.cpp
    sources/framecapture.cpp
    sources/facedetection_yunet.cpp
    sources/facerecognition_sface.cpp
)

# Create executable
add_executable(fdr ${SOURCES})

# Link libraries
target_link_libraries(fdr
    ${OpenCV_LIBS}
    pthread
)

# Copy model files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models/ DESTINATION ${CMAKE_BINARY_DIR}/models/)
