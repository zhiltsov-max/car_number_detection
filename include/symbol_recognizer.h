#ifndef SYMBOL_RECOGNIZER_H
#define SYMBOL_RECOGNIZER_H

#include <vector>
#include "opencv2\core\core.hpp"
#include "opencv2\ml\ml.hpp"


class SymbolRecognizer {
public:
    typedef int SymbolClass;
    static const SymbolClass UNRECOGNIZED_SYMBOL;

    void load(const char* fileName);
    void train(const cv::Mat& trainData, const cv::Mat& appearances, const char* outputfileName);
    void prepareTrainData(const cv::Mat& trainData, cv::Mat& out);

    SymbolClass recognizeSymbol(const cv::Mat& symbol);

    struct SymbolInfo {
        char repr;
    };
    const SymbolInfo& getSymbolInfo(const SymbolClass& class_);    
    void addSymbolInfo(const SymbolClass& class_, const SymbolInfo& info);
    void setClassCount(size_t count);
private:
    static const cv::Size2i FRAME_SIZE;
    static const double SYMBOL_ACCEPT_THRESHOLD;

    struct Features;
    void extractFeatures(const cv::Mat& img, Features& features);
    void findHistograms(const cv::Mat& img, Features& features);
 
    typedef std::vector<SymbolInfo> Classes;
    Classes classes;

    typedef cv::NeuralNet_MLP ANN;
    ANN recognizer;
};

#endif // SYMBOL_RECOGNIZER_H
