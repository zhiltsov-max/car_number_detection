#pragma once
#include <string>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\ml\ml.hpp"
#include <fstream>

class PlateSVM
{
    cv::SVMParams params;
    cv::SVM svm;
    std::vector<cv::String> positives;
    std::vector<cv::String> negatives;
public: 
    PlateSVM(void);
    ~PlateSVM(void){}
    void getNegatives(const cv::String &path);
    void getPositives(const cv::String &path);
    void train();
    void init(const std::string& trainDataPath);
    std::vector<cv::Mat> predict(std::vector<cv::Mat> &plates);
};

