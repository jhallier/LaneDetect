#include "LaneDetect.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

/* Constructor. Calculates the transformation matrix from image (src) to warped (dst) space. M is the transformation matrix from image to warped space
Minv is the inverse, transforming back from warped to image space. To achieve this, 4 points in the source space are mapped to 4 points in the destination space. The 4 points were taken by experiment from the autonomous vehicle simulation Carla (www.carla.org)*/
LaneDetect::LaneDetect() {

    cv::Point2f src_data[4];
    src_data[0] = cv::Point2f(396, 324);
    src_data[1] = cv::Point2f(431, 324);
    src_data[2] = cv::Point2f(689, 599);
    src_data[3] = cv::Point2f(249, 599);
    cv::Point2f dst_data[4];
    dst_data[0] = cv::Point2f(300, 0);
    dst_data[1] = cv::Point2f(499, 0);
    dst_data[2] = cv::Point2f(499, 599);
    dst_data[3] = cv::Point2f(300, 599);

    this->M = cv::getPerspectiveTransform(src_data, dst_data);
    this->Minv = cv::getPerspectiveTransform(dst_data, src_data);
}

LaneDetect::~LaneDetect() {}

cv::Mat LaneDetect::canny_edge_detection(cv::Mat img, double canny_thresh[2]) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // Canny edge detection
    cv::Mat edges;
    int apertureSize = 3;
    bool L2gradient = true;
    cv::Canny(gray, edges, canny_thresh[0], canny_thresh[1], apertureSize, L2gradient);

    return edges;
}

std::vector<tk::spline> LaneDetect::edge_find_fit(cv::Mat edges){

    /* Output type of reduce operation needs to be explicitly defined */
    cv::Mat histogram(edges.rows, 1, CV_32SC1);
    /* Calculates sum along axis 0 (reduces to column vector) */
    cv::reduce(edges, histogram, 0, cv::REDUCE_SUM, CV_32SC1);
    int sww = 15; // half the search window width
    int nwindows = 20; // Vertical res / nwindows should divide w/o remainder
    int swh = static_cast<int>(edges.rows / nwindows); // Search window height
    int diff_lane_x = static_cast<int>(edges.rows / 16); // How far away do lane markers need to be to belong to two different lanes?
    std::vector<int> lane_x_positions = {};
    // Mask all values that exceed mask value to find all lanes in the image
    double max_vert_pixels;
    cv::minMaxIdx(histogram, NULL, &max_vert_pixels, NULL, NULL);
    int mask_value = static_cast<int>(max_vert_pixels * 0.1);
    // Create a mask where only values greater than the mask value are retained
    cv::Mat hist_mask(histogram.rows, histogram.cols, CV_8U);
    cv::inRange(histogram, mask_value, max_vert_pixels, hist_mask);

    /* Finds all peaks in the histogram, each one indicating a different lane in the image */
    while (true) {
        // Mask the original histogram
        cv::Mat masked_hist;
        histogram.copyTo(masked_hist, hist_mask);
        double current_peak;
        cv::Point2i current_peak_pos;
        cv::minMaxLoc(masked_hist, NULL, &current_peak, NULL, &current_peak_pos);
        lane_x_positions.push_back(current_peak_pos.x);
        int left = 0;
        if ((current_peak_pos.x - diff_lane_x) > 0) {
            left = current_peak_pos.x - diff_lane_x;
        }
        int right = edges.cols - 1;
        if ((current_peak_pos.x + diff_lane_x) < right) {
            right = current_peak_pos.x + diff_lane_x;
        }
        /* Set all values left and right of selected peak to zero. hist_mask is a column vector, so rows=1, colums = image width */
        set_Mat_value(hist_mask, 0, 0, left, right, 0);
        /* Exit loop if hist_mask is empt (all peaks found), otherwise continue with next peak */
        if (cv::countNonZero(hist_mask) == 0) {
            break;
        }
    }

    std::sort(lane_x_positions.begin(), lane_x_positions.end());
    std::vector<tk::spline> lane_fits;

    for (int x_marker: lane_x_positions) {
        std::vector<double> pts_x, pts_y;
        std::vector<cv::Rect> rectangles;

        for (int i = 0; i < nwindows; ++i) {
            /* Looks for all non-zero pixel values in a search window and find the center */
            int swup = i*swh;
            int swdown = (i+1) * swh;
            int swleft = x_marker - sww;
            int swright = x_marker + sww;
            cv::Mat search_window = edges(cv::Range(swup, swdown), cv::Range(swleft, swright));
            cv::Rect rect_coords(swleft, swup, 2*sww, swh);
            rectangles.push_back(rect_coords);

            cv::Mat nonZero;
            cv::findNonZero(search_window, nonZero);
            /* Calculate mean x and y value of all non-zero pixels in the search windows */
            std::vector<int> x_vals = {}, y_vals = {};
            for (uint j = 0; j < nonZero.total(); ++j) {
                int x = nonZero.at<cv::Point>(j).x;
                x_vals.push_back(x);
                int y = nonZero.at<cv::Point>(j).y;
                y_vals.push_back(y);
            }
            /* If there are nonzero values, calculate the mean (x,y) */
            if (nonZero.size().height > 0) {
                int x_sum = 0, y_sum = 0;
                for (uint j = 0; j < x_vals.size(); ++j) {
                    x_sum += x_vals[j];
                    y_sum += y_vals[j];
                }
                double x_mean = x_sum / static_cast<double>(x_vals.size()) + x_marker - sww;
                double y_mean = y_sum / static_cast<double>(y_vals.size()) + swup;
                pts_x.push_back(x_mean);
                pts_y.push_back(y_mean);
            }
        }

        #if defined(DEBUG)
            debug_warped_image(edges, pts_x, pts_y, rectangles);
        #endif

        /* polynomial fit only possible with at least 3 (x,y) pairs */
        if (pts_x.size() >= 3) {
            /* Performs a 3rd degree polynomial spline fit */
            tk::spline curve_fit;
            #if defined(DEBUG)
                std::cout << "Spline fit on points: ";
                for (int i = 0; i < pts_y.size(); ++i) {
                    std::cout << "(" << pts_y[i] << "," << pts_x[i] << "),";
                }
                std::cout << std::endl;
            #endif
            curve_fit.set_points(pts_y, pts_x);
            lane_fits.push_back(curve_fit);
        }
        /* Raise error condition */
        else {

        }
    }
    return lane_fits;
}

/* Sets values in cv::Mat mat2d from (row_beg, col_beg) to (row_end, col_end) to value */
void LaneDetect::set_Mat_value(cv::Mat &mat2d, int row_beg, int row_end, int col_beg, int col_end, int value) {

    #if defined(DEBUG)
        int maskNonZero = cv::countNonZero(mat2d);
        std::cout << "Nonzero elements in mask before transformation: " << maskNonZero << std::endl;
    #endif

    for (int i = row_beg; i <= row_end; ++i) {
        for (int j = col_beg; j <= col_end; ++j) {
            mat2d.at<unsigned char>(i, j) = value;
        }
    }

    #if defined(DEBUG)
        std::cout << "Removing elements from " << row_beg << "," << col_beg << " to " << row_end << "," << col_end << std::endl;
        maskNonZero = cv::countNonZero(mat2d);
        std::cout << "Nonzero elements in mask after transformation: " << maskNonZero << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    #endif
}

cv::Mat LaneDetect::plot_lane_image(cv::Mat img, std::vector<tk::spline> lane_fits) {

    /* Create an image to draw the lines on */
    cv::Mat draw_img(img.rows, img.cols, CV_8UC3);
    /* Generate x and y values for plotting */
    std::vector<int> ploty = {};
    for (int i = 0; i <= img.rows; i+= (img.rows / 5)) {
        ploty.push_back(i);
    }
    cv::Scalar colors[] = {{0,255,0},{0,0,255},{255,0,0},{128,128,128},{255,0,255},{0,255,255},{255,255,0}};
    int col_ind = 0;

    while (lane_fits.size() > 1) {
        std::vector<std::vector<cv::Point>> polygons = {};
        tk::spline left = lane_fits[0];
        tk::spline right = lane_fits[1];
        /* Creates a vector with all points for a polygon. Starts at the top with the left and right points and appends alternatively to the front and the back of the vector, so that the resulting polygon is continuous */
        std::vector<cv::Point> one_polygon = {};

        for (uint i = 0; i < ploty.size(); ++i) {
            /* Evaluates x values for left lane at y position */
            int y = ploty[i];
            one_polygon.push_back(cv::Point(left(y), y));
        }
        for (int i = ploty.size() - 1; i >= 0; --i) {
            /* Evaluates x values for right lane at y position */
            int y = ploty[i];
            one_polygon.push_back(cv::Point(right(y), y));
        }
        polygons.push_back(one_polygon);
        /* Draw the lane onto the warped blank image */
        cv::fillPoly(draw_img, polygons, colors[col_ind], cv::LINE_8, 0);
        
        col_ind++;
        lane_fits.erase(lane_fits.begin());
    }

    #if defined(DEBUG)
        cv::namedWindow("DEBUG: Warped Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("DEBUG: Warped Image", draw_img);
        cv::waitKey(0);
    #endif
        
    /* Warp the blank back to original image space using inverse perspective matrix (Minv) */
    cv::Mat draw_warp;
    cv::warpPerspective(draw_img, draw_warp, this->Minv, draw_img.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    /* Combine the result with the original image */
    cv::Mat result;
    cv::addWeighted(img, 1, draw_warp, 0.5, 0, result);
    
    return result;
}

void LaneDetect::debug_warped_image(cv::Mat image, std::vector<double> pts_x, std::vector<double> pts_y, std::vector<cv::Rect> search_windows) {
    for (uint i = 0; i < pts_x.size(); ++i) {
        int x = static_cast<int>(pts_x[i]);
        int y = static_cast<int>(pts_y[i]);
        cv::Point2i center(x,y);
        std::cout << "Plotting center at " << center << std::endl;
        cv::drawMarker(image, center, cv::Scalar(255,255,255),cv::MARKER_TILTED_CROSS, 10, 1, 8);
    }
    for (uint i = 0; i < search_windows.size(); ++i) {
        cv::rectangle(image, search_windows[i], cv::Scalar(255,255,255), 1, 8, 0);
    }
    cv::namedWindow("DEBUG Lane with markers", cv::WINDOW_AUTOSIZE);
    cv::imshow("DEBUG Lane with markers", image);
    cv::waitKey(0);
}

/* Processes a single image
Inputs: cv::Mat img - single image in OpenCV Mat format
Outputs: output - single image with detected lane lines in OpenCV Mat format */
cv::Mat LaneDetect::process_single_image(cv::Mat img) {

    double canny_thresh[2] = {50, 255};

    cv::Mat warped;
    cv::warpPerspective(img, warped, this->M, img.size());

    cv::Mat edges = canny_edge_detection(warped, canny_thresh);

    std::vector<tk::spline> lane_fits = edge_find_fit(edges);
    cv::Mat output = plot_lane_image(img, lane_fits);

    return output;
}
