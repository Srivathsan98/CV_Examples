# // File: CMakeLists.txt
# cmake_minimum_required(VERSION 3.10)
# project(FaceRecognitionApp)

# find_package(OpenCV REQUIRED)

# add_executable(FaceRecognitionApp
#     main.cpp
#     framecapture.cpp
#     facedetector.cpp
#     facerecognizer.cpp
# )

# include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(FaceRecognitionApp ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.10)
project(FaceDetectionRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/includes
)

# Source files
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/sources/*.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Output path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

