#include "symbol_recognizer.h"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"


const double SymbolRecognizer::SYMBOL_ACCEPT_THRESHOLD = 0.7;
const SymbolRecognizer::SymbolClass SymbolRecognizer::UNRECOGNIZED_SYMBOL = -1;
const cv::Size2i SymbolRecognizer::FRAME_SIZE (10, 10);

struct SymbolRecognizer::Features {
    cv::Mat hhist;
    cv::Mat vhist;
    cv::Mat imgSample;
    cv::Mat completed;
};

SymbolRecognizer::SymbolClass SymbolRecognizer::recognizeSymbol(const cv::Mat& symbol) {
    cv::Mat img;
    cv::resize(symbol, img, FRAME_SIZE, 0, 0, CV_INTER_AREA);

    Features symbolFeatures;
    extractFeatures(img, symbolFeatures);

    float class_id = recognizer.predict(symbolFeatures.completed);
    return (int)class_id;
}

const SymbolRecognizer::SymbolInfo& SymbolRecognizer::getSymbolInfo(const SymbolRecognizer::SymbolClass& class_) {
    return classes[class_];
}

void SymbolRecognizer::findHistograms(const cv::Mat& img, SymbolRecognizer::Features& features) {
	CV_Assert(features.vhist.cols == img.cols && features.vhist.type() == CV_32F);
	CV_Assert(features.hhist.cols == img.rows && features.hhist.type() == CV_32F);

    for (int i = 0; i < img.cols; ++i) {
        features.vhist.at<float>(i) = (float)cv::countNonZero(img.col(i));
    }

    for (int i = 0; i < img.rows; ++i) {
        features.hhist.at<float>(i) = (float)cv::countNonZero(img.row(i));
    }

    // normalize
    features.vhist *= 1.f / img.cols;
    features.hhist *= 1.f / img.rows;
}

void SymbolRecognizer::extractFeatures(const cv::Mat& img, Features& features) {
    CV_Assert(img.rows == FRAME_SIZE.height && img.cols == FRAME_SIZE.width);
    // Warning! Be careful with indicies.
    features.completed.create(1,img.rows + img.cols + FRAME_SIZE.area(), CV_32F);
    features.hhist = features.completed(cv::Range(0, 1), cv::Range(0, img.rows));
    features.vhist = features.completed(cv::Range(0, 1), cv::Range(img.rows, img.rows + img.cols));
    findHistograms(img, features);
        
    cv::resize(img, features.imgSample, FRAME_SIZE, 0, 0, CV_INTER_AREA); //maybe Lanczos
    features.imgSample = features.imgSample.reshape(0, 1);
	features.imgSample.convertTo(features.completed(cv::Range(0, 1), cv::Range(features.hhist.total() + features.vhist.total(), features.completed.cols)), CV_32F);
}

void SymbolRecognizer::setClassCount(size_t count) {
    classes.resize(count);
}

void SymbolRecognizer::addSymbolInfo(const SymbolClass& class_, const SymbolInfo& info) {
    classes[class_] = info;
}

void SymbolRecognizer::train(const cv::Mat& trainData, const cv::Mat& appearances, const char* outputFileName) {
	cv::SVMParams params;
	recognizer.train(trainData, appearances.reshape(0, appearances.cols), cv::Mat(), cv::Mat(), params);
	recognizer.save(outputFileName);
}

void SymbolRecognizer::load(const char* fileName) {
    recognizer.load(fileName);
}

void SymbolRecognizer::prepareTrainData(const cv::Mat& trainData, cv::Mat& out) {
    out = cv::Mat();
    for (int row = 0; row < trainData.rows; ++row) {
        Features features;
        cv::Mat img = trainData.row(row).reshape(0, 10);
        extractFeatures(img, features);
        out.push_back(features.completed);
    }
}