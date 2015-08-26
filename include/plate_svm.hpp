#ifndef PLATE_SVM_H
#define PLATE_SVM_H

#include <string>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\ml\ml.hpp"
#include <fstream>

class PlateSVM
{
public: 
    PlateSVM(void);
    ~PlateSVM(void){}
    void getNegatives(cv::String& path);
    void getPositives(cv::String& path);
    void train();
    std::vector<cv::Mat> predict(std::vector<cv::Mat>& plates);

private:
    cv::SVMParams params;
    cv::SVM svm;
    std::vector<cv::String> positives;
    std::vector<cv::String> negatives;
};

#endif // PLATE_SVM_H