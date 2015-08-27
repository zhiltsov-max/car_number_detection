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

    cv::Mat probabilities;
    recognizer.predict(symbolFeatures.completed, probabilities);

    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(probabilities, 0, &maxVal, 0, &maxLoc);
	maxVal /= 2 * 1.7159; /*read docs*/
	maxVal += 0.5;
	std::cout << "Max prob: " << maxVal << " at " << maxLoc.x << std::endl;
    if (maxVal < SYMBOL_ACCEPT_THRESHOLD) {
        return UNRECOGNIZED_SYMBOL;
    }
    return maxLoc.x;
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
    features.completed.create(1,/* img.rows + img.cols + */FRAME_SIZE.area(), CV_32F);
    //features.hhist = features.completed(cv::Range(0, 1), cv::Range(0, img.rows));
    //features.vhist = features.completed(cv::Range(0, 1), cv::Range(img.rows, img.rows + img.cols));
    //findHistograms(img, features);
        
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
    cv::Mat layerSizes(1, 30, CV_32S);
    layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = 10; //magic
	layerSizes.at<int>(2) = 10; //magic
	layerSizes.at<int>(3) = 10; //magic
	layerSizes.at<int>(4) = 10; //magic
	layerSizes.at<int>(5) = 10; //magic
	layerSizes.at<int>(6) = 10; //magic
	layerSizes.at<int>(7) = 10; //magic
	layerSizes.at<int>(8) = 10; //magic
	layerSizes.at<int>(9) = 10; //magic
	layerSizes.at<int>(10) = 10; //magic
	layerSizes.at<int>(11) = 10; //magic
	layerSizes.at<int>(12) = 10; //magic
	layerSizes.at<int>(13) = 10; //magic
	layerSizes.at<int>(14) = 10; //magic
	layerSizes.at<int>(15) = 10; //magic
	layerSizes.at<int>(16) = 10; //magic
	layerSizes.at<int>(17) = 10; //magic
	layerSizes.at<int>(18) = 10; //magic
	layerSizes.at<int>(19) = 10; //magic
	layerSizes.at<int>(20) = 10; //magic
	layerSizes.at<int>(21) = 10; //magic
	layerSizes.at<int>(22) = 10; //magic
	layerSizes.at<int>(23) = 10; //magic
	layerSizes.at<int>(24) = 10; //magic
	layerSizes.at<int>(25) = 10; //magic
	layerSizes.at<int>(26) = 10; //magic
	layerSizes.at<int>(27) = 10; //magic
	layerSizes.at<int>(28) = 10; //magic
    layerSizes.at<int>(29) = (int)classes.size(); //0.94% errors...
    recognizer.create(layerSizes, CvANN_MLP::SIGMOID_SYM);

    cv::Mat trainClasses(trainData.rows, classes.size(), CV_32F, cv::Scalar(0));
    for (int i = 0; i < trainClasses.rows; ++i) {
        trainClasses.at<float>(i, appearances.at<int>(i)) = 1.f;
    }

    cv::Mat weights(1, trainData.rows, CV_32F, cv::Scalar::all(1));

	CvANN_MLP_TrainParams trainParams;
	trainParams.train_method = CvANN_MLP_TrainParams::BACKPROP;
    recognizer.train(trainData, trainClasses, weights, cv::Mat(), trainParams);
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