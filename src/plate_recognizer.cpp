#include "plate_recognizer.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

#if defined(_DEBUG_)
    #include <iostream>
#endif


static const double RECOGNIZER_THRESHOLD = 60.0;
static const double RECOGNIZER_THRESHOLD_MAX = 255.0;

static const double RECOGNIZER_SYMBOL_ASPECT_RATIO = 45.0 / 77.0;
static const double RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR = 0.35;
static const double RECOGNIZER_SYMBOL_MIN_ASPECT_RATIO = 0.2;

TNumberPlateDetector::Recognizer::SymbolParameters::SymbolParameters() :
    minHeight(15), 
    maxHeight(28),
    acceptedError(RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR),
    minAspectRatio(RECOGNIZER_SYMBOL_MIN_ASPECT_RATIO),
    aspectRatio(RECOGNIZER_SYMBOL_ASPECT_RATIO),
    maxUsedAreaPercent(0.8)
{}


TNumberPlateDetector::Number TNumberPlateDetector::Recognizer::recognizeNumber(const cv::Mat& plate) {
    CV_Assert(plate.type() == CV_8UC1);

    cv::Mat img_thresh;
    cv::threshold(plate, img_thresh, RECOGNIZER_THRESHOLD, RECOGNIZER_THRESHOLD_MAX, CV_THRESH_BINARY_INV);

    cv::blur(img_thresh, img_thresh, cv::Size(3, 3));

#if defined(_DEBUG_)
    cv::imshow("Thresh", img_thresh);
#endif

    cv::Mat img_contours = img_thresh.clone();
    std::vector<std::vector< cv::Point >> contours;
    cv::findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

#if defined(_DEBUG_)
    std::cout << "Found contours: " << contours.size() << std::endl;
    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        cv::Scalar color(255);
        cv::drawContours(img_contours, contours, it - contours.cbegin(), color);
    }
    cv::imshow("Contours", img_contours);
    cv::waitKey();
#endif

    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        cv::Rect symbolBounds = cv::boundingRect(*it);
        cv::Mat symbol = img_thresh(
            cv::Range(symbolBounds.y, symbolBounds.y + symbolBounds.height),
            cv::Range(symbolBounds.x, symbolBounds.x + symbolBounds.width)            
            ).clone();
        if (verifySymbolSize(symbol) == true) {
            symbolFrames.emplace_back(symbol);
        }
    }

#if defined(_DEBUG_)
    std::cout << "Found contours: " << symbolFrames.size() << std::endl;
    for (auto it = symbolFrames.cbegin(), iend = symbolFrames.cend(); it != iend; ++it) {
        cv::imshow(std::to_string(it - symbolFrames.cbegin()), *it);
    }
    cv::waitKey();
#endif

    return Number();
}

bool TNumberPlateDetector::Recognizer::verifySymbolSize(const cv::Mat& bounds) {
    double charAspect = (double)bounds.cols / (double)bounds.rows;
    double maxAspectRatio = symbolParameters.aspectRatio + symbolParameters.aspectRatio * symbolParameters.acceptedError;
    double usedArea = cv::countNonZero(bounds);
    double fullArea = bounds.cols * bounds.rows;
    double percent = usedArea / fullArea;
    if ((percent < symbolParameters.maxUsedAreaPercent) &&
        (symbolParameters.minAspectRatio < charAspect) && (charAspect < maxAspectRatio) &&
        (symbolParameters.minHeight <= bounds.rows) && (bounds.rows < symbolParameters.maxHeight)
       )
    {
        return true;
    } else {
        return false;
    }
}
