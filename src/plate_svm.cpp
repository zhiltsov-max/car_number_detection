#include "plate_svm.hpp"


using namespace cv;


PlateSVM::PlateSVM(void) {
    params.svm_type = cv::SVM::C_SVC;
    params.kernel_type = cv::SVM::LINEAR;
}

void PlateSVM::getPositives(cv::String& path) {
    std::ifstream info(path);
    if (info.is_open() == false) {
        std::cerr << "Failed to open file 'positives/names.txt'." << std::endl;
        return;
    }
    while (info.good() == true) {
        std::string img_name;
        std::getline(info, img_name);
        positives.push_back("positives/"+img_name);
    }
    info.close();
}

void PlateSVM::getNegatives(cv::String &path) {
    std::ifstream info(path);
    if (info.is_open() == false) {
        std::cerr << "Failed to open file 'negatives/names.txt'." << std::endl;
        return;
    }
    while (info.good() == true) {
        std::string img_name;
        std::getline(info, img_name);
        negatives.push_back("negatives/"+img_name);
    }
    info.close();
}

void PlateSVM::train() {
    int width = 144, height = 33;
    cv::Mat trainData(positives.size() + negatives.size(), width * height, CV_32F);
    for (size_t i = 0; i < positives.size(); ++i)
    {
        cv::Mat image = imread(positives[i], CV_LOAD_IMAGE_GRAYSCALE);
        image.convertTo(image, CV_32F);
        CV_Assert(image.rows == height);
        CV_Assert(image.cols == width);
        image.reshape(0, 1).copyTo(trainData.row(i));
    }
    for (size_t i = 0; i < negatives.size(); ++i)
    {
        cv::Mat image = imread(negatives[i], CV_LOAD_IMAGE_GRAYSCALE);
        image.convertTo(image, CV_32F);
        CV_Assert(image.rows == height);
        CV_Assert(image.cols == width);
        image.reshape(0, 1).copyTo(trainData.row(i + positives.size()));
    }

    cv::Mat classes(positives.size() + negatives.size(), 1, CV_32F, 0.0f);
    classes.rowRange(0, positives.size()).setTo(1.0f);
    
    svm.train(trainData, classes, Mat(), Mat(), params);
    svm.save("plates_svm_classifier.yml");
}

std::vector<cv::Mat> PlateSVM::predict(std::vector<cv::Mat>& plates)
{
    vector<cv::Mat> true_plates;
    for (size_t i = 0; i < plates.size(); ++i)
        {
            cv::Mat plate = plates[i];
            imshow("plate?", plate);
            waitKey();
            plate.convertTo(plate, CV_32F);
            plate = plate.reshape(0, 1);
            float class_id = svm.predict(plate);
            if (class_id == 1.0f) {
                std::cout<<"plate";
                true_plates.push_back(plate);
                // It's a plate.
            } else {
                std::cout<<"not a plate";
                // It's not a plate.
            }
        }
}