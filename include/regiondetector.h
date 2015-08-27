#ifndef REGION_DETECTOR_H
#define REGION_DETECTOR_H

#include "opencv2/core/core.hpp"
#include <iostream>
#include <vector>


class RegionDetector {
public:
    void threshold(const cv::Mat& img);
	void close();
    void getMask();
    std::vector<cv::Mat> proceed(const cv::Mat& src);

private:
    std::vector<cv::RotatedRect> rects;

    cv::Mat histeq(const cv::Mat& src);
    bool verifySizes(const cv::RotatedRect& rect);
    void preprocess(const cv::Mat& src, cv::Mat& out);
    std::vector<cv::Mat> floodFillMask(const cv::Mat& src, cv::Mat& working);
    void getContours(const cv::Mat& src, cv::Mat& working);

};

#endif // REGION_DETECTOR_H