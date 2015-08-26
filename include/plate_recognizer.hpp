#ifndef PLATE_RECOGNIZER_H
#define PLATE_RECOGNIZER_H

#include "number_plate_detector.hpp"
#include "opencv2\core\core.hpp"
#include "symbol_recognizer.h"


class TNumberPlateDetector::Recognizer {
public:
    TNumberPlateDetector::Number recognizeNumber(const cv::Mat& plate);

    struct PlateParameters {
        double groupAppearanceThreshold;

        typedef std::vector<cv::Rect> Groups;
        Groups groups;
        
        struct SymbolParameters {
            double minHeight;
            double maxHeight;
            double acceptedError;
            double aspectRatio;
            double maxUsedAreaPercent;

            SymbolParameters();    
        };

        SymbolParameters symbolParameters;
        
        static PlateParameters* RUSSIAN_;
        static const PlateParameters& RUSSIAN();
    };
    const PlateParameters& getPlateParameters() const;
    PlateParameters& getPlateParameters();

    void train();
private:
    PlateParameters plateParameters;

    typedef size_t SymbolGroup;

    struct SymbolFrame {
        cv::Mat frame;
        cv::Rect position;
        SymbolGroup group;
    };
    typedef std::vector<SymbolFrame> SymbolFrames;
    
    SymbolGroup determineSymbolGroup(const cv::Rect& position, const cv::Size& plateSize);

    bool verifySymbolFrame(const SymbolFrame& frame, const cv::Mat& plate);
    bool verifySymbolSize(const cv::Mat& bounds);
    bool verifySymbolPosition(const cv::Rect& position, const cv::Size& plateSize);

    SymbolRecognizer symbolRecognizer;
};

#endif // PLATE_RECOGNIZER_H
