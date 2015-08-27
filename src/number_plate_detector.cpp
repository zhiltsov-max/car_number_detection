#include "number_plate_detector.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "plate_recognizer.hpp"
#include "plate_svm.hpp"
#include "regiondetector.h"


TNumberPlateDetector::Number::operator std::string() const {
    std::string result;
    for (auto it = cbegin(), iend = cend(); it != iend; ++it) {
        result.append(*it);
    }
    return result;
}

TNumberPlateDetector::Number TNumberPlateDetector::getNumber(const cv::Mat& frame) {
    
    RegionDetector det;
    std::vector<cv::Mat> plates;
    /*plates = det.proceed(frame);

    PlateSVM psvm;
    psvm.init("detector/detector.yml");
    plates = psvm.predict(plates);
    */
    cv::Mat  monochrome;
    cv:cvtColor(frame, monochrome, CV_BGR2GRAY);
    plates.push_back(monochrome);

    Recognizer recognizer;
    recognizer.init();

    recognizer.getPlateParameters() = Recognizer::PlateParameters::RUSSIAN();
    for (auto it = plates.cbegin(), iend = plates.cend(); it != iend; ++it) {
        Number number = recognizer.recognizeNumber(*it);
        std::cout << "Found number: " << static_cast<std::string>(number) << std::endl;
    }
    return Number();
}

void TNumberPlateDetector::train() {
    Recognizer recognizer;
    recognizer.train();
}
