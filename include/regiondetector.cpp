#include "regiondetector.h"


bool RegionDetector::verifySizes(RotatedRect mr){  
    float error = 0.4; 
    //Car plate size: 52x11 aspect 4,7272 =======TODO: change size=====
    float aspect = 4.7272; 
    //Set a min and max roi. All other patchs are discarded 
    int min = 15*aspect*15; // minimum roi 
    int max = 125*aspect*125; // maximum roi 
    //Get only patchs that match to a respect ratio. 
    float rmin = aspect-aspect*error; 
    float rmax = aspect+aspect*error; 

    int roi = mr.size.height * mr.size.width; 
    float k = (float)mr.size.width / (float)mr.size.height; 
    if(k<1)
        k = (float)mr.size.height / (float)mr.size.width; 
 
    if(( roi < min || roi > max ) || ( k < rmin || k > rmax )){ 
        return false; 
    }else{ 
        return true; 
    } 
} 


RegionDetector::RegionDetector(void)
{
}
void RegionDetector::threshold(Mat src)
{
	//convert to gray
	Mat img_gray;
	cv::cvtColor(src,img_gray, CV_BGR2GRAY);
    imshow("gray", img_gray);
	blur(img_gray, img_gray, Size(5,5));
	imshow("blur", img_gray);

    //find first horizontal derivative( vertical lines which car plate usually has)
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1,0);
	imshow("sobel", img_sobel);

    //threshold
	cv::threshold(img_sobel,img_temp, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    imshow("thrs", img_temp);
}
void RegionDetector::close()
{
	//apply morphological close
	Mat element = getStructuringElement(MORPH_RECT, Size(17,3));
	morphologyEx(img_temp, img_temp, CV_MOP_CLOSE, element);
    imshow("close", img_temp);
}
void RegionDetector::getMask()
{
    //find contours
	vector <vector<Point> > contours;
    cv::findContours(img_temp,contours,
	                    CV_RETR_EXTERNAL,	//retrive external contours
                        CV_CHAIN_APPROX_NONE);	// all pixels of each contour
    //extract rectangle of minimal roi
    //Start to iterate to each contour found
    vector <vector<Point> >::iterator itc = contours.begin();
    vector<RotatedRect> rects;
					
    //Remove patch that has no inside limits of aspect ratio and roi
    while (itc!=contours.end())
    {
    //create bounding rect of object
	    RotatedRect rr= minAreaRect(Mat(*itc));
	    if (!verifySizes(rr)){
		    itc= contours.erase(itc);
	    }else{
		    ++itc;
		    rects.push_back(rr);
	    }
    }

    // Draw blue contours on a white image 
    Mat result; 
    img_temp.copyTo(result); 
    cv::drawContours(result,contours, 
            -1, // draw all contours 
             cv::Scalar(255,0,0), // in blue 
             1); // with a thickness of 1
    imshow("nn",result);

}

