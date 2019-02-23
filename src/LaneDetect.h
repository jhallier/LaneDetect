#include <opencv2/core/core.hpp>
#include "./spline/src/spline.h"
#include <vector>

/* Prints out more detailed debug messages. To de/-activate, #undef/#define DEBUG */
#undef DEBUG

class LaneDetect {

private:
    cv::Mat M, Minv;

public:

    LaneDetect();
    ~LaneDetect();

    cv::Mat canny_edge_detection(cv::Mat img, double canny_thresh[2]);

    std::vector<tk::spline> edge_find_fit(cv::Mat edges);

    cv::Mat plot_lane_image(cv::Mat img, std::vector<tk::spline> lane_fits);

    cv::Mat process_single_image(cv::Mat img);

    void set_Mat_value(cv::Mat &mat2d, int row_beg, int row_end, int col_beg, int col_end, int value);

    void debug_warped_image(cv::Mat image, std::vector<double> pts_x, std::vector<double> pts_y, std::vector<cv::Rect> search_windows);
    
};