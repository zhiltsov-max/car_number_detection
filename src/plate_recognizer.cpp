#include "plate_recognizer.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

#if defined(_DEBUG_)
    #include <iostream>
#endif


static const double RECOGNIZER_THRESHOLD = 60.0;
static const double RECOGNIZER_THRESHOLD_MAX = 255.0;

static const double RECOGNIZER_SYMBOL_ASPECT_RATIO = 45.0 / 77.0;
static const double RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR = 0.35;
static const double RECOGNIZER_SYMBOL_MIN_ASPECT_RATIO = 0.2;

TNumberPlateDetector::Recognizer::PlateParameters russian_init() {
    TNumberPlateDetector::Recognizer::PlateParameters pp;
    pp.groupAppearanceThreshold = 0.8;

    // GOST 50577-93
    pp.groups.push_back(cv::Rect(30, 68, 58, 42));
}
const TNumberPlateDetector::Recognizer::PlateParameters::RUSSIAN = russian_init();

TNumberPlateDetector::Recognizer::PlateParameters::SymbolParameters::SymbolParameters() :
    minHeight(15), 
    maxHeight(28),
    acceptedError(RECOGNIZER_SYMBOL_ACCEPTED_ASPECT_ERROR),
    minAspectRatio(RECOGNIZER_SYMBOL_MIN_ASPECT_RATIO),
    aspectRatio(RECOGNIZER_SYMBOL_ASPECT_RATIO),
    maxUsedAreaPercent(0.8)
{}

TNumberPlateDetector::Number TNumberPlateDetector::Recognizer::recognizeNumber(const cv::Mat& plate) {
    CV_Assert(plate.type() == CV_8UC1);

    cv::Mat img_thresh;
    cv::threshold(plate, img_thresh, RECOGNIZER_THRESHOLD, RECOGNIZER_THRESHOLD_MAX, CV_THRESH_BINARY_INV);

    cv::blur(img_thresh, img_thresh, cv::Size(3, 3));

#if defined(_DEBUG_)
    cv::imshow("Thresh", img_thresh);
#endif

    cv::Mat img_contours = img_thresh.clone();
    std::vector<std::vector< cv::Point >> contours;
    cv::findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

#if defined(_DEBUG_)
    std::cout << "Found contours: " << contours.size() << std::endl;
    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        cv::Scalar color(255);
        cv::drawContours(img_contours, contours, it - contours.cbegin(), color);
    }
    cv::imshow("Contours", img_contours);
    cv::waitKey();
#endif

    SymbolFrames symbolFrames;

    for (auto it = contours.cbegin(), iend = contours.cend(); it != iend; ++it) {
        SymbolFrame frame;
        frame.position = cv::boundingRect(*it);
        frame.frame = img_thresh(
            cv::Range(frame.position.y, frame.position.y + frame.position.height),
            cv::Range(frame.position.x, frame.position.x + frame.position.width)            
            ).clone();

        if (verifySymbolFrame(frame, img_thresh) == true) {
            frame.group = determineSymbolGroup(frame.position, cv::Size(img_thresh.cols, img_thresh.rows));

            symbolFrames.push_back(frame);
        }
    }

#if defined(_DEBUG_)
    //std::cout << "Found contours: " << symbolFrames.size() << std::endl;
    //for (auto it = symbolFrames.cbegin(), iend = symbolFrames.cend(); it != iend; ++it) {
    //    cv::imshow(std::to_string(it - symbolFrames.cbegin()), *it);
    //}
    //cv::waitKey();
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
    return true; //TO DO:
}

bool TNumberPlateDetector::Recognizer::verifySymbolSize(const cv::Mat& bounds) {
    const auto& symbolParameters = plateParameters.symbolParameters;

    double charAspect = (double)bounds.cols / (double)bounds.rows;
    double maxAspectRatio = symbolParameters.aspectRatio + symbolParameters.aspectRatio * symbolParameters.acceptedError;
    double usedArea = cv::countNonZero(bounds);
    double fullArea = bounds.cols * bounds.rows;
    double percent = usedArea / fullArea;

    return 
       (
        (percent < symbolParameters.maxUsedAreaPercent) &&
        (symbolParameters.minAspectRatio < charAspect) && (charAspect < maxAspectRatio) &&
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
        int result= recognizer.recognizeSymbol(samples.row(i));
        if (result != classes.at<int>(i)) {
            errors++;
        }
    }
    return errors / samples.rows;
}

void TNumberPlateDetector::Recognizer::train() {
    cv::Mat trainData, trainClasses, data, classes, samples, samplesClasses;
    cv::FileStorage fs;
    fs.open("recognizer_train_data.xml", cv::FileStorage::READ);
    fs["data"] >> data;

    int trainDataSize = 100;
    int trainDataBegin = rand() % (data.rows - trainDataSize);
    trainData = data(cv::Range(trainDataBegin, trainDataBegin + trainDataSize), cv::Range(0, data.cols)).clone();
    samples = data(cv::Range(0, trainDataBegin), cv::Range(0, data.cols)).clone();
    samples.push_back(data(cv::Range(trainDataBegin + trainDataSize, data.rows), cv::Range(0, data.cols)));

    fs["classes"] >> classes;
    trainClasses = data(cv::Range(trainDataBegin, trainDataBegin + trainDataSize), cv::Range(0, data.cols)).clone();
    samplesClasses = data(cv::Range(0, trainDataBegin), cv::Range(0, data.cols)).clone();
    samplesClasses.push_back(data(cv::Range(trainDataBegin + trainDataSize, data.rows), cv::Range(0, data.cols)));

    symbolRecognizer.train(trainData, trainClasses, "recognizer_trained.xml");
    
    std::cout << "Trained with error: " << test(samples, samplesClasses, symbolRecognizer) << std::endl;
}
