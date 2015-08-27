#ifndef PLATE_SVM_H
#define PLATE_SVM_H

#include <string>
#include "opencv2\core\core.hpp"
#include "opencv2\ml\ml.hpp"

class PlateSVM
{
public: 
    PlateSVM();

    void init(const std::string& trainDataPath);
    void getNegatives(const cv::String& path);
    void getPositives(const cv::String& path);
    void train();
    std::vector<cv::Mat> predict(const std::vector<cv::Mat>& plates);

private:
    cv::SVMParams params;
    cv::SVM svm;
    std::vector<cv::String> positives;
    std::vector<cv::String> negatives;
};

#endif // PLATE_SVM_H