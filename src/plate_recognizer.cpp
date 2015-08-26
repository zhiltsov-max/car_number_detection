#include "plate_recognizer.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <fstream>

#if defined(_DEBUG_)
    #include <iostream>
#endif


static const double RECOGNIZER_THRESHOLD = 60.0;
static const double RECOGNIZER_THRESHOLD_MAX = 255.0;

static const double RECOGNIZER_SYMBOL_ASPECT_RATIO = 45.0 / 77.0;
static const double RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR = 0.35;

TNumberPlateDetector::Recognizer::PlateParameters* TNumberPlateDetector::Recognizer::PlateParameters::RUSSIAN_ = nullptr;
const TNumberPlateDetector::Recognizer::PlateParameters& TNumberPlateDetector::Recognizer::PlateParameters::RUSSIAN() {
    if (RUSSIAN_ == nullptr) { 
        RUSSIAN_ = new TNumberPlateDetector::Recognizer::PlateParameters();
        auto& pp = *RUSSIAN_;
        pp.groupAppearanceThreshold = 0.8;

        // GOST 50577-93
        pp.groups.push_back(cv::Rect(30, 34, 42, 58)); // first letter
        pp.groups.push_back(cv::Rect(82, 16, 42, 76)); // first digit
        pp.groups.push_back(cv::Rect(134, 16, 42, 76)); // second digit
        pp.groups.push_back(cv::Rect(186, 16, 42, 76)); // third digit
        pp.groups.push_back(cv::Rect(238, 34, 42, 58)); // second letter
        pp.groups.push_back(cv::Rect(285, 34, 42, 58)); // third letter
        pp.groups.push_back(cv::Rect(380, 13, 42, 58)); // region #1
        pp.groups.push_back(cv::Rect(427, 13, 42, 58)); // region #2
        pp.groups.push_back(cv::Rect(474, 13, 42, 58)); // region #3

        pp.size = cv::Size(520, 112);

        pp.symbolParameters.acceptedError = RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR;
        pp.symbolParameters.aspectRatio = 42.0 / 76.0;
        pp.symbolParameters.minHeight = 42.0 * 0.7;
        pp.symbolParameters.maxHeight = 42.0 * 1.3;
        pp.symbolParameters.maxUsedAreaPercent = 0.8;
    }
    return *RUSSIAN_;
}

TNumberPlateDetector::Recognizer::PlateParameters::SymbolParameters::SymbolParameters() :
    minHeight(15), 
    maxHeight(28),
    acceptedError(RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR),
    aspectRatio(RECOGNIZER_SYMBOL_ASPECT_RATIO),
    maxUsedAreaPercent(0.8)
{}

void TNumberPlateDetector::Recognizer::preprocessImage(const cv::Mat& plate, cv::Mat& out) {
    cv::Mat img_thresh;
    cv::threshold(plate, img_thresh, RECOGNIZER_THRESHOLD, RECOGNIZER_THRESHOLD_MAX, CV_THRESH_BINARY_INV);
    cv::blur(img_thresh, img_thresh, cv::Size(3, 3));

    cv::Size size(img_thresh.cols, img_thresh.rows);
    if (size != plateParameters.size) {
        if (size.area() < plateParameters.size.area()) {
            cv::resize(img_thresh, img_thresh, plateParameters.size, 0, 0, CV_INTER_CUBIC);
        } else {
            cv::resize(img_thresh, img_thresh, plateParameters.size, 0, 0, CV_INTER_AREA);
        }
    }

#if defined(_DEBUG_)
    cv::imshow("Thresh", img_thresh);
#endif

    out = img_thresh;
}

TNumberPlateDetector::Number TNumberPlateDetector::Recognizer::recognizeNumber(const cv::Mat& plate) {
    CV_Assert(plate.type() == CV_8UC1);

    cv::Mat img;
    preprocessImage(plate, img);

    cv::Mat img_contours = img.clone();
    std::vector<std::vector< cv::Point >> contours;
    cv::findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

#if defined(_DEBUG_)
    std::cout << "Found contours: " << contours.size() << std::endl;
    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        cv::Scalar color(255);
        cv::drawContours(img_contours, contours, it - contours.cbegin(), color);
    }
    cv::imshow("Contours", img_contours);
    //cv::waitKey();
#endif

    SymbolFrames symbolFrames;

    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        SymbolFrame frame;
        frame.position = cv::boundingRect(*it);
        frame.frame = img(
            cv::Range(frame.position.y, frame.position.y + frame.position.height),
            cv::Range(frame.position.x, frame.position.x + frame.position.width)            
            ).clone();

        if (verifySymbolFrame(frame, img) == true) {
            frame.group = determineSymbolGroup(frame.position, cv::Size(img.cols, img.rows));
            if (plateParameters.groups.size() <= frame.group) {
                continue;
            }

            symbolFrames.push_back(frame);
        }
    }

#if defined(_DEBUG_)
    std::cout << "Found contours: " << symbolFrames.size() << std::endl;
    for (auto it = symbolFrames.cbegin(), iend = symbolFrames.cend(); it != iend; ++it) {
        cv::imshow(std::to_string(it - symbolFrames.cbegin()), *it);
    }
    cv::waitKey();
#endif

    Number number;
    number.resize(plateParameters.groups.size());

    for (auto it = symbolFrames.cbegin(), iend = symbolFrames.cend(); it != iend; ++it) {
        SymbolRecognizer::SymbolClass category = symbolRecognizer.recognizeSymbol((*it).frame);
        if (category != SymbolRecognizer::UNRECOGNIZED_SYMBOL) {
            number[(*it).group].push_back(symbolRecognizer.getSymbolInfo(category).repr);
        }
    }
    
    return number;
}

TNumberPlateDetector::Recognizer::SymbolGroup TNumberPlateDetector::Recognizer::determineSymbolGroup(const cv::Rect& position, const cv::Size& plateSize) {
    for (auto it = plateParameters.groups.cbegin(), iend = plateParameters.groups.cend(); it != iend; ++it) {
        double metric = (double)((*it) & position).area() / (double)((*it) | position).area();
        if (plateParameters.groupAppearanceThreshold < metric) {
            return it - plateParameters.groups.cbegin();
        }
    }
    return plateParameters.groups.size() + 1;
}

bool TNumberPlateDetector::Recognizer::verifySymbolFrame(const SymbolFrame& frame, const cv::Mat& plate) {    
    if (verifySymbolPosition(frame.position, cv::Size(plate.cols, plate.rows)) == false) {
        return false;
    }

    if (verifySymbolSize(frame.frame) == false) {
        return false;
    }

    return true;
}

bool TNumberPlateDetector::Recognizer::verifySymbolPosition(const cv::Rect& position, const cv::Size& plateSize) {
    for (auto it = plateParameters.groups.cbegin(), iend = plateParameters.groups.cend(); it != iend; ++it) {
        if ( ((*it) & position).area() != 0 ) {
            return true;
        }
    }
    return false;
}

bool TNumberPlateDetector::Recognizer::verifySymbolSize(const cv::Mat& bounds) {
    const auto& symbolParameters = plateParameters.symbolParameters;

    double charAspect = (double)bounds.cols / (double)bounds.rows;
    double minAspectRatio = symbolParameters.aspectRatio * symbolParameters.acceptedError;
    double maxAspectRatio = symbolParameters.aspectRatio * (1.0 + symbolParameters.acceptedError);
    double usedArea = cv::countNonZero(bounds);
    double fullArea = bounds.cols * bounds.rows;
    double percent = usedArea / fullArea;

    return 
       (
        (percent < symbolParameters.maxUsedAreaPercent) &&
        (minAspectRatio < charAspect) && (charAspect < maxAspectRatio) &&
        (symbolParameters.minHeight <= bounds.rows) && (bounds.rows < symbolParameters.maxHeight)
       );
}

const TNumberPlateDetector::Recognizer::PlateParameters& TNumberPlateDetector::Recognizer::getPlateParameters() const {
    return plateParameters;
}

TNumberPlateDetector::Recognizer::PlateParameters& TNumberPlateDetector::Recognizer::getPlateParameters() {
    return plateParameters;
}

float test(const cv::Mat& samples, const cv::Mat& classes, SymbolRecognizer& recognizer) {
    float errors = 0;
    for (int i = 0; i < samples.rows; ++i) {
        int result = recognizer.recognizeSymbol(samples.row(i).reshape(0, 10));
        if (result != classes.at<int>(i)) {
            errors++;
        }
    }
    return errors / samples.rows;
}

void TNumberPlateDetector::Recognizer::train() {
    const int classCount = 23;
    cv::Mat trainData, trainClasses, data, classes, samples, samplesClasses;

    std::ifstream info("recognizer/data.txt");
    if (info.is_open() == false) {
        std::cerr << "Failed to open file 'recognizer/data.txt'." << std::endl;
        return;
    }

    int i = 0;
    while (info.good() == true) {
        std::string img_name;
        std::string img_class_;
        std::getline(info, img_name, ' ');
        std::getline(info, img_class_);

        cv::Mat img = cv::imread("recognizer/" + img_name, CV_LOAD_IMAGE_GRAYSCALE);
        if (img.empty() == false) {
            data.push_back(img.reshape(0, 1));
            
            int img_class = std::atoi(img_class_.c_str());

            cv::Mat class_(1, classCount, CV_32S, cv::Scalar(0));
            if (0 <= img_class ) {
                class_.at<int>(img_class) = 1;
            }
            classes.push_back(class_);
        }
    }
    info.close();

    int trainDataSize = 1000;
    int trainDataBegin = rand() % (data.rows - trainDataSize);
    symbolRecognizer.prepareTrainData(data(cv::Range(trainDataBegin, trainDataBegin + trainDataSize), cv::Range(0, data.cols)).clone(), trainData);
    samples = data(cv::Range(0, trainDataBegin), cv::Range(0, data.cols)).clone();
    samples.push_back(data(cv::Range(trainDataBegin + trainDataSize, data.rows), cv::Range(0, data.cols)));

    trainClasses = classes(cv::Range(trainDataBegin, trainDataBegin + trainDataSize), cv::Range(0, classes.cols)).clone();
    samplesClasses = classes(cv::Range(0, trainDataBegin), cv::Range(0, classes.cols)).clone();
    samplesClasses.push_back(classes(cv::Range(trainDataBegin + trainDataSize, classes.rows), cv::Range(0, classes.cols)));

    symbolRecognizer.setClassCount(classCount);

    char symbols[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y'};
    SymbolRecognizer::SymbolInfo symbolInfo;
    for (int i = 0; i < classCount; ++i) {
        symbolInfo.repr = symbols[i];
        symbolRecognizer.addSymbolInfo(0, symbolInfo);
    }
    symbolRecognizer.train(trainData, trainClasses, "recognizer_trained.xml");
    
    std::cout << "Trained with error: " << test(samples, samplesClasses, symbolRecognizer) << std::endl;
}
