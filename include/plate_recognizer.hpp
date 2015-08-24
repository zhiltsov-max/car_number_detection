#ifndef PLATE_RECOGNIZER_H
#define PLATE_RECOGNIZER_H

#include "number_plate_detector.hpp"
#include "opencv2\core\core.hpp"

class TNumberPlateDetector::Recognizer {
public:
    TNumberPlateDetector::Number recognizeNumber(const cv::Mat& plate);

    struct SymbolParameters {
        double minHeight;
        double maxHeight;
        double acceptedError;
        double aspectRatio;
        double minAspectRatio;
        double maxUsedAreaPercent;

        SymbolParameters();    
    };

private:
    SymbolParameters symbolParameters;

    typedef std::vector<cv::Mat> SymbolFrames;
    SymbolFrames symbolFrames;

    bool verifySymbolSize(const cv::Mat& bounds);
};

#endif // PLATE_RECOGNIZER_H