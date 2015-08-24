#ifndef SYMBOL_RECOGNIZER_H
#define SYMBOL_RECOGNIZER_H

#include <vector>
#include "opencv2\core\core.hpp"


class SymbolRecognizer {
public:
    typedef int SymbolClass;

    SymbolClass recognizeSymbol(const cv::Mat& symbol);
private:
    static const cv::Size2i FRAME_SIZE;

    struct Features;
    void extractFeatures(const cv::Mat& img, Features& features);
    void findHistograms(const cv::Mat& img, Features& features);
};

#endif // SYMBOL_RECOGNIZER_H
