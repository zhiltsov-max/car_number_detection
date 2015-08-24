#include "symbol_recognizer.h"
#include "opencv2\imgproc\imgproc.hpp"


struct SymbolRecognizer::Features {
    cv::Mat hhist;
    cv::Mat vhist;
    cv::Mat imgSample;
    cv::Mat completed;
};


const cv::Size2i SymbolRecognizer::FRAME_SIZE (5, 5);

SymbolRecognizer::SymbolClass SymbolRecognizer::recognizeSymbol(const cv::Mat& symbol) {
    Features symbolFeatures;
    extractFeatures(symbol, symbolFeatures);

    std::vector<double> appearanceProbabilities;

}

void SymbolRecognizer::findHistograms(const cv::Mat& img, SymbolRecognizer::Features& features) {
    features.vhist.create(1, img.cols, CV_32F);
    features.hhist.create(1, img.rows, CV_32F);

    for (int i = 0; i < img.cols; ++i) {
        features.vhist.at<float>(i) = cv::countNonZero(img.col(i));
    }

    for (int i = 0; i < img.rows; ++i) {
        features.hhist.at<float>(i) = cv::countNonZero(img.row(i));
    }

    // normalize
    // TO DO: do in other way, divide to MAX_VALUE
    double min = 0.0;
    double max = 0.0;
    cv::minMaxLoc(features.vhist, &min, &max);
    features.vhist *= 1.0 / (max - min);

    cv::minMaxLoc(features.hhist, &min, &max);
    features.hhist *= 1.0 / (max - min);
}

void SymbolRecognizer::extractFeatures(const cv::Mat& img, Features& features) {
    // Warning! Be careful with indicies.
    features.completed.create(1, img.rows + img.cols + FRAME_SIZE.area(), CV_32F);
    features.hhist = features.completed(cv::Range(0, 1), cv::Range(0, img.rows));
    features.vhist = features.completed(cv::Range(0, 1), cv::Range(img.rows, img.rows + img.cols));
    findHistograms(img, features);

    features.imgSample;
    cv::resize(img, features.imgSample, FRAME_SIZE, 0, 0, CV_INTER_AREA); //maybe Lanczos
    features.imgSample.copyTo(features.completed(cv::Range(0, 1), cv::Range(img.rows + img.cols, features.completed.cols)));
}
