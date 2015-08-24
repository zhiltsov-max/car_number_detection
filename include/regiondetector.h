#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <iostream>
/*
//Algorithm


*/
using namespace cv;
class RegionDetector
{
	Mat img_temp;
    Mat histeq(Mat src);
    bool verifySizes(RotatedRect rr);
public:

	RegionDetector(void);
	virtual ~RegionDetector(void){}
    void threshold(Mat src);
	void close();
    void getMask();
};

