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
    bool showimage;
	Mat img_temp;
    Mat src;
    vector <RotatedRect> rects;

    Mat histeq(Mat src);
    bool verifySizes(RotatedRect rr);
    void preprocessing();
    void getContours();
    vector <Mat> floodfillmask();
public:
	RegionDetector(void);
    RegionDetector(bool si);
	virtual ~RegionDetector(void){}
    vector <Mat> proceed(Mat _src);
};

