#include "regiondetector.h"
#include <ctime>


bool RegionDetector::verifySizes(const RotatedRect& rect){
    float error = 0.4;
    //Car plate size: 520x112 aspect 4,7272
    float aspect = 4.7272;
    //Set a min and max area. All other patchs are discarded
    int min = 15*aspect*15; // minimum area
    int max = 125*aspect*125; // maximum area
    //Get only patchs that match to a respect ratio.
    float rmin = aspect-aspect*error;
    float rmax = aspect+aspect*error;

    int roi = rect.size.height * rect.size.width;
    float k = (float)rect.size.width / (float)rect.size.height;
    if(k<1)
        k = (float)rect.size.height / (float)rect.size.width;

    if(( roi < min || roi > max ) || ( k < rmin || k > rmax )){
        return false;
    }else{
        return true;
    }
}
Mat RegionDetector::histeq(Mat input)
{
    Mat out(input.size(), input.type());
    if(input.channels()==3){
        Mat hsv;
        vector<Mat> hsvSplit;
        cvtColor(input, hsv, CV_BGR2HSV);
        split(hsv, hsvSplit);
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, hsv);
        cvtColor(hsv, out, CV_HSV2BGR);
    }else if(input.channels()==1){
        equalizeHist(input, out);
    }
    return out;
}


RegionDetector::RegionDetector(bool si = true)
{
    showimage = si;
}
void RegionDetector::preprocessing()
{
    Mat img_gray;
    cv::cvtColor(src,img_gray, CV_BGR2GRAY);
    blur(img_gray, img_gray, Size(5,5));
    if (showimage)
        imshow("blur", img_gray);
    //Р“В­Р“В Р“ВµР“В®Р“В¤Р“РЃР“В¬ Р“В±Р“Р„Р“В®Р“Р‡Р“В«Р“ТђР“В­Р“РЃР“С— Р“Р€Р“В®Р“В°Р“РЃР“В§Р“В®Р“В­Р“Р†Р“В Р“В«Р“СР“В­Р“В»Р“Вµ Р“В«Р“РЃР“В­Р“РЃР“В©
    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1,0);
    if (showimage)
        imshow("sobel", img_sobel);
    //threshold
    Mat img_threshold;
    cv::threshold(img_sobel,img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    if (showimage)
        imshow("threshold", img_threshold);
    //Р“В¬Р“В®Р“В°Р“Т‘Р“В®Р“В«Р“В®Р“Р€Р“РЃР“В·Р“ТђР“В±Р“Р„Р“В®Р“Тђ Р“В§Р“В Р“Р„Р“В°Р“В»Р“Р†Р“РЃР“Тђ
    Mat element = getStructuringElement(MORPH_RECT, Size(23,5));
    morphologyEx(img_threshold, img_temp, CV_MOP_CLOSE, element);
    if (showimage)
        imshow("close", img_temp);
}
void RegionDetector::getContours()
{
    //Р“В­Р“В Р“ВµР“В®Р“В¤Р“РЃР“В¬ Р“Р„Р“В®Р“В­Р“Р†Р“С–Р“В°Р“В» Р“В­Р“В  Р“СћР“В®Р“В§Р“В¬Р“В®Р“В¦Р“В­Р“В»Р“Вµ Р“В§Р“В­Р“В Р“Р„Р“В Р“Вµ
    vector <vector<Point> > contours;
    cv::findContours(   img_temp,
                        contours,
                        CV_RETR_EXTERNAL,	//extreme outer contours
                        CV_CHAIN_APPROX_NONE);	// Р“СћР“В±Р“Тђ Р“Р‡Р“РЃР“Р„Р“В±Р“РЃР“В«Р“РЃ Р“Р„Р“В Р“В¦Р“В¤Р“В®Р“Р€Р“В® Р“Р„Р“В®Р“В­Р“Р†Р“С–Р“В°Р“В 
    //Start to iterate to each contour found
    vector <vector<Point> >::iterator itc = contours.begin();

    //Remove patch that has no inside limits of aspect ratio and area
    while (itc!=contours.end())
    {
        //create bounding rect of object
        RotatedRect rr = minAreaRect(Mat(*itc));
        if (!verifySizes(rr)){
            itc = contours.erase(itc);
        }else{
            itc++;
            rects.push_back(rr);
        }
    }
    // Draw blue contours on a white image
    src.copyTo(img_temp);
    cv::drawContours(img_temp,contours,
            -1,  // draw all contours
             cv::Scalar(255,0,0),// in blue
             1); // thickness
    if (showimage)
        imshow("contours",img_temp);
}
vector <Mat> RegionDetector::floodfillmask()
{
    vector <Mat> plates;
    for(int i=0; i< rects.size(); i++)
    {
        //For better rect cropping for each posible box
        //Make floodfill algorithm because the plate has white background
        //And then we can retrieve more clearly the contour box
        circle(img_temp, rects[i].center, 3, Scalar(0,255,0), -1);
        float minSize=(rects[i].size.width < rects[i].size.height)?rects[i].size.width:rects[i].size.height;
        minSize=minSize-minSize*0.5;
        srand ( time(NULL) );
        //floodfill Р“Р‡Р“В Р“В°Р“В Р“В¬Р“ТђР“Р†Р“В°Р“В»
        Mat mask;
        mask.create(src.rows + 2, src.cols + 2, CV_8UC1);
        mask= Scalar::all(0);
        int connectivity = 4;
        int newMaskVal = 255;
        int loDiff = 30;
        int upDiff = 30;
        int NumSeeds = 10;
        Rect ccomp;
        int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
        for(int j=0; j<NumSeeds; j++)
        {
            Point seed;
            seed.x=rects[i].center.x+rand()%(int)minSize-(minSize/2);
            seed.y=rects[i].center.y+rand()%(int)minSize-(minSize/2);
            circle(img_temp, seed, 1, Scalar(0,255,255), -1);
            int area = floodFill(  src,
                                   mask,
                                   seed,
                        Scalar(255,0,0),
                                 &ccomp,
           Scalar(loDiff,loDiff,loDiff),
           Scalar(upDiff,upDiff,upDiff),
                                  flags);
        }
        if (showimage)
            imshow("MASK", mask);

        vector<Point> pointsInterest;
        Mat_<uchar>::iterator itMask = mask.begin<uchar>();
        Mat_<uchar>::iterator end = mask.end<uchar>();
        while(itMask!=end)
        {
            if(*itMask==255)
                pointsInterest.push_back(itMask.pos());
            itMask++;
        }
        RotatedRect minRect = cv::minAreaRect(pointsInterest);
        if(verifySizes(minRect))
        {
            // Р“В°Р“РЃР“В±Р“С–Р“ТђР“В¬ Р“В·Р“ТђР“Р†Р“В»Р“В°Р“ТђР“ВµР“С–Р“Р€Р“В®Р“В«Р“СР“В­Р“РЃР“Р„Р“РЃ
            Point2f rect_points[4]; minRect.points(rect_points);
            for( int j = 0; j < 4; j++ )
            {
                line(img_temp, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255));
            }
            if (showimage)
                imshow("red",img_temp);
            //Р“В¬Р“В Р“Р†Р“В°Р“РЃР“В¶Р“В  Р“Р‡Р“В®Р“СћР“В®Р“В°Р“В®Р“Р†Р“В 
            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            if(r<1)
                angle = 90+angle;
            Mat rotmat = getRotationMatrix2D(minRect.center, angle,1);

            //Р“Р‡Р“В®Р“СћР“В®Р“В°Р“В Р“В·Р“РЃР“СћР“В Р“ТђР“В¬ Р“Р„Р“В Р“В°Р“Р†Р“РЃР“В­Р“Р„Р“С–
            Mat img_rotated;
            warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);

            //Р“СћР“В»Р“В°Р“ТђР“В§Р“В Р“ТђР“В¬ Р“Р„Р“С–Р“В±Р“В®Р“Р„
            Size rect_size=minRect.size;
            if(r < 1)
                std::swap(rect_size.width, rect_size.height);
            Mat img_crop;
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

            Mat resultResized;
            resultResized.create(33,144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

            //Р“СћР“В»Р“В°Р“В Р“СћР“В­Р“РЃР“СћР“В Р“ТђР“В¬ Р“РЃР“В§Р“В®Р“РЋР“В°Р“В Р“В¦Р“ТђР“В­Р“РЃР“Тђ
            Mat grayResult;
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);
            blur(grayResult, grayResult, Size(3,3));
            grayResult=histeq(grayResult);

            plates.push_back(grayResult);
            if (showimage)
                imshow("Contours", grayResult);
        }
    }
    return plates;
}

vector<Mat> RegionDetector::proceed(Mat _src)
{
    src = _src;
    preprocessing();
    getContours();
    return floodfillmask();
}
