#include "regiondetector.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <ctime>


using  namespace cv;

bool RegionDetector::verifySizes(const RotatedRect& rect) {
    double error = 0.4;
    //Car plate size: 520x112 aspect 4,7272
    double aspect = 4.7272;
    //Set a min and max area. All other patchs are discarded
    int min = (int)15*aspect*15; // minimum area
    int max = (int)125*aspect*125; // maximum area
    //Get only patchs that match to a respect ratio.
    double rmin = aspect-aspect*error;
    double rmax = aspect+aspect*error;

    int roi = rect.size.height * rect.size.width;
    double k = (double)rect.size.width / (double)rect.size.height;
    if (k < 1) {
        k = (double)rect.size.height / (double)rect.size.width;
    }
    if (( roi < min || roi > max ) || ( k < rmin || k > rmax )){
        return false;
    } else {
        return true;
    }
}

Mat RegionDetector::histeq(const Mat& input) {
    Mat out(input.size(), input.type());
    if(input.channels()==3){
        Mat hsv;
        vector<Mat> hsvSplit;
        cvtColor(input, hsv, CV_BGR2HSV);
        split(hsv, hsvSplit);
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, hsv);
        cvtColor(hsv, out, CV_HSV2BGR);
    } else if (input.channels()==1)  {
        equalizeHist(input, out);
    }
    return out;
}


void RegionDetector::preprocess(const cv::Mat& src, cv::Mat& out) {
    Mat img_gray;
    cv::cvtColor(src, img_gray, CV_BGR2GRAY);
    blur(img_gray, img_gray, Size(5,5));

#if defined(_DEBUG_)
    imshow("blur", img_gray);
#endif

    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1,0);

#if defined(_DEBUG_)
    imshow("sobel", img_sobel);
#endif

    //threshold
    Mat img_threshold;
    cv::threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

#if defined(_DEBUG_)
    imshow("threshold", img_threshold);
#endif

    Mat element = getStructuringElement(MORPH_RECT, Size(23,5));
    morphologyEx(img_threshold, out, CV_MOP_CLOSE, element);

#if defined(_DEBUG_)
    imshow("close", out);
#endif
}

void RegionDetector::getContours(const cv::Mat& src, cv::Mat& working) {
    vector <vector<Point> > contours;
    cv::findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    //Remove patch that has no inside limits of aspect ratio and area
    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        RotatedRect bounds = minAreaRect(Mat(*it));
        if (verifySizes(bounds) == true) {            
            rects.push_back(bounds);
        }
#if defined(_DEBUG_)
        else {
            it = contours.erase(it);
        }
#endif
    }

#if defined(_DEBUG_)
    src.copyTo(working);
    cv::drawContours(working, contours, CV_FILLED, cv::Scalar(255, 0, 0), 1);

    imshow("contours", working);
#endif
}

vector<Mat> RegionDetector::floodFillMask(const cv::Mat& src, cv::Mat& working) {
    vector<Mat> plates;
    srand(time(0));
    for(int i = 0; i < rects.size(); i++) {
        //For better rect cropping for each posible box
        //Make floodfill algorithm because the plate has white background
        //And then we can retrieve more clearly the contour box
#if defined(_DEBUG_)
        circle(working, rects[i].center, 3, Scalar(0,255,0), -1);
#endif

        float minSize = 
            (rects[i].size.width < rects[i].size.height) ? 
            rects[i].size.width : 
            rects[i].size.height;

        minSize = minSize - minSize * 0.5;

        Mat mask(working.rows + 2, working.cols + 2, CV_8UC1, Scalar::all(0));

        int connectivity = 4;
        int newMaskVal = 255;
        int loDiff = 30;
        int upDiff = 30;
        int NumSeeds = 10;

        Rect ccomp;
        int flags = connectivity | (newMaskVal << 8) | CV_FLOODFILL_FIXED_RANGE | CV_FLOODFILL_MASK_ONLY;
        
        for(int j = 0; j < NumSeeds; j++) {
            Point seed(rects[i].center.x+rand() % (int)minSize-(minSize/2),
                       rects[i].center.y+rand() % (int)minSize-(minSize/2)
                      );

#if defined(_DEBUG_)
            circle(working, seed, 1, Scalar(0,255,255), -1);
#endif

            floodFill(src, mask, seed, Scalar(255,0,0), &ccomp,
                      Scalar(loDiff, loDiff, loDiff),
                      Scalar(upDiff, upDiff, upDiff),
                      flags
                    );
        }

#if defined(_DEBUG_)
        imshow("MASK", mask);
#endif

        vector<Point> pointsInterest;
        for (auto it = mask.begin<uchar>(), iend = mask.end<uchar>(); it != iend; ++it)  {
            if (*it == 255) {
                pointsInterest.push_back(it.pos());
            }
        }

        RotatedRect minRect = cv::minAreaRect(pointsInterest);
        if (verifySizes(minRect) == true) {
            Point2f rect_points[4]; 
            minRect.points(rect_points);

            for( int j = 0; j < 4; j++ ) {
                line(working, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255));
            }

#if defined(_DEBUG_)
            imshow("red", working);
#endif

            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            if (r < 1.f) {
                angle = 90.f + angle;
            }
            Mat rotmat = getRotationMatrix2D(minRect.center, angle,1);

            Mat img_rotated;
            warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);

            Size rect_size = minRect.size;
            if (r < 1.f) {
                std::swap(rect_size.width, rect_size.height);
            }
            Mat img_crop;
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

            Mat resultResized;
            resultResized.create(33, 144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

            Mat grayResult;
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);
            blur(grayResult, grayResult, Size(3, 3));
            grayResult = histeq(grayResult);

            plates.push_back(grayResult);

#if defined(_DEBUG_)
            imshow("Contours", grayResult);
#endif
        }
    }
    return plates;
}

vector<Mat> RegionDetector::proceed(const Mat& src) {
    cv::Mat working;
    preprocess(src, working);
    getContours(src, working);
    return floodFillMask(src, working);
}
