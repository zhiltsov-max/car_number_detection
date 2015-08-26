#include "symbol_recognizer.h"
#include "opencv2\imgproc\imgproc.hpp"


const double SymbolRecognizer::SYMBOL_ACCEPT_THRESHOLD = 0.75;
const SymbolRecognizer::SymbolClass SymbolRecognizer::UNRECOGNIZED_SYMBOL = -1;
const cv::Size2i SymbolRecognizer::FRAME_SIZE (10, 10);

struct SymbolRecognizer::Features {
    cv::Mat hhist;
    cv::Mat vhist;
    cv::Mat imgSample;
    cv::Mat completed;
};

SymbolRecognizer::SymbolClass SymbolRecognizer::recognizeSymbol(const cv::Mat& symbol) {
    Features symbolFeatures;
    extractFeatures(symbol, symbolFeatures);

    cv::Mat probabilities;
    recognizer.predict(symbolFeatures.completed, probabilities);
    
    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(probabilities, 0, &maxVal, 0, &maxLoc);
    if (maxVal < SYMBOL_ACCEPT_THRESHOLD) {
        return UNRECOGNIZED_SYMBOL;
    }
    return maxLoc.x;
}

const SymbolRecognizer::SymbolInfo& SymbolRecognizer::getSymbolInfo(const SymbolRecognizer::SymbolClass& class_) {
    return classes[class_];
}

void SymbolRecognizer::findHistograms(const cv::Mat& img, SymbolRecognizer::Features& features) {
    features.vhist.create(1, img.cols, CV_32F);
    features.hhist.create(1, img.rows, CV_32F);

    for (int i = 0; i < img.cols; ++i) {
        features.vhist.at<float>(i) = (float)cv::countNonZero(img.col(i));
    }

    for (int i = 0; i < img.rows; ++i) {
        features.hhist.at<float>(i) = (float)cv::countNonZero(img.row(i));
    }

    // normalize
    double min = 0.0;
    double max = 255.0; //0.0;
    //cv::minMaxLoc(features.vhist, &min, &max);
    features.vhist *= 1.0 / (max - min);

    max = 255.0;
    //cv::minMaxLoc(features.hhist, &min, &max);
    features.hhist *= 1.0 / (max - min);
}

void SymbolRecognizer::extractFeatures(const cv::Mat& img, Features& features) {
    // Warning! Be careful with indicies.
    features.completed.create(1, img.rows + img.cols + FRAME_SIZE.area(), CV_32F);
    features.hhist = features.completed(cv::Range(0, 1), cv::Range(0, img.rows));
    features.vhist = features.completed(cv::Range(0, 1), cv::Range(img.rows, img.rows + img.cols));
    findHistograms(img, features);
        
    cv::resize(img, features.imgSample, FRAME_SIZE, 0, 0, CV_INTER_AREA); //maybe Lanczos
    features.imgSample = features.imgSample.reshape(0, 1);
    features.imgSample.copyTo(features.completed(cv::Range(0, 1), cv::Range(features.hhist.total() + features.vhist.total(), features.completed.cols)));
}

void SymbolRecognizer::setClassCount(size_t count) {
    classes.resize(count);
}

void SymbolRecognizer::addSymbolInfo(const SymbolClass& class_, const SymbolInfo& info) {
    classes[class_] = info;
}

void SymbolRecognizer::train(const cv::Mat& trainData, const cv::Mat& appearances, const char* outputFileName) {
    cv::Mat layerSizes(1, 3, CV_32S);
    layerSizes.at<int>(0) = trainData.cols;
    layerSizes.at<int>(1) = 20; //magic
    layerSizes.at<int>(2) = (int)classes.size();
    recognizer.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1, 1);

    cv::Mat trainClasses(trainData.rows, classes.size(), CV_32F);
    for (int i = 0; i < trainClasses.rows; ++i) {
        for (int k = 0; k < trainClasses.cols; ++k) {
            if (k == appearances.at<int>(i)) {
                trainClasses.at<float>(i,k) = 1.f;
            } else {
                trainClasses.at<float>(i,k) = 0.f;
            }
        }
    }
    cv::Mat weights(1, trainData.rows, CV_32F, cv::Scalar::all(1));
    recognizer.train(trainData, trainClasses, weights);
    recognizer.save(outputFileName);
}

void SymbolRecognizer::load(const char* fileName) {
    recognizer.load(fileName);
}

void SymbolRecognizer::prepareTrainData(const cv::Mat& trainData, cv::Mat& out) {
    out = cv::Mat();
    for (int row = 0; row < trainData.rows; ++row) {
        Features features;
        features.imgSample = trainData.row(row).reshape(0, 10);
        extractFeatures(features.imgSample, features);
        out.push_back(features.completed);
    }
}