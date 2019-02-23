#include "LaneDetect.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int argc, char** argv) {
    cv::Mat image;
    std::string filename;

    if (argc != 2) {
        std::cout << "No image provided, using standard test_image.jpg. Syntax for using your own image: ./LaneDetect [filename]" << std::endl;
        filename = "../images/test_image.jpg";
    }
    else {
        filename = argv[1];
    }

    image = cv::imread(filename, cv::IMREAD_COLOR);
    if (!image.data) {
        std::cout << "Error. Image data corrupted." << std::endl;
        return -1;
    }

    cv::Mat processed_image;
    LaneDetect LaneD;

    processed_image = LaneD.process_single_image(image);

    cv::imwrite("../images/detected_lanes.jpg", processed_image);

    cv::namedWindow("Lane detection result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Lane detection result", processed_image);
    cv::waitKey(0);
   
    return 0;
}