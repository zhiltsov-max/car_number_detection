#include "number_plate_detector.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "plate_recognizer.hpp"


TNumberPlateDetector::Number::operator std::string() const {
    std::string result;
    for (auto it = cbegin(), iend = cend(); it != iend; ++it) {
        result.append(*it);
    }
    return result;
}

TNumberPlateDetector::Number TNumberPlateDetector::getNumber(const cv::Mat& frame) {
    cv::Mat  monochrome;
    cv:cvtColor(frame, monochrome, CV_BGR2GRAY);
    Recognizer recognizer;
    recognizer.getPlateParameters() = Recognizer::PlateParameters::RUSSIAN();
    return recognizer.recognizeNumber(monochrome);    
}

void TNumberPlateDetector::train() {
    Recognizer recognizer;
    recognizer.train();
}
