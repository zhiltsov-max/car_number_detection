#include <iostream>
#include <string>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\ml\ml.hpp"
#include <fstream>
#include "PlateSVM.h"
//#include "number_plate_detector.hpp"

#include "regiondetector.h"

/* Algorithm
I. Plate detection
1. Segmentation
2. Classification 
II. Plate recognition
1. Segmentation
2. Classification
*/

void printHelp() {
    std::cout << 
        "\tUsage: <image> [-t, --train <classifierName>; -h, --help]" << std::endl <<
        std::endl <<
        "-t <classifier name>: Train a specified classifier" <<std::endl;
}

static const char* params = 
    "{ 1 | image     |     | Path to target image    }"
    "{ h | help      |false| Print help              }"
    "{ t | train     |false| Train classifier <name> }"
    "{ p | pos       |     | Positives for training  }"
    "{ n | neg       |     | Negatives for training  }";

int main(int argc, char* argv[]) {
	cv::CommandLineParser parser(argc, argv, params);
	std::string imagePath = parser.get<std::string>("image");
    std::string classifierName = parser.get<std::string>("train");
    std::string positivesNamesPath = parser.get<std::string>("pos");
    std::string negativesNamesPath = parser.get<std::string>("neg");
	
    if (classifierName.empty() && imagePath.empty()) {
        printHelp();
        return 0;
    }
    if (positivesNamesPath.empty() || negativesNamesPath.empty()){
        std::cerr << "Positives or negatives path missing" << std::endl;
    }
    if (classifierName.empty() == false) {
        // TO DO:
    }

    if (imagePath.empty() == false) {
        cv::Mat image = cv::imread(imagePath);
        CV_Assert(image.empty() == false);

        //TNumberPlateDetector detector;
        
        std::cout << "Image: " << imagePath << std::endl;
        
        PlateSVM psvm;
        psvm.getNegatives(negativesNamesPath);
        psvm.getPositives(positivesNamesPath);
        psvm.train();

        RegionDetector det(false);
        Mat tmp = imread(imagePath);
        vector <Mat> plates;
        plates = det.proceed(tmp);
        psvm.predict(plates);
        
        while (1)
        {
            int k = waitKey(10);
            if (k == 27)break;            
        }
        return 0;
    }

	return 0;
}