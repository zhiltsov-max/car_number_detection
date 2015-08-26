#include <iostream>
#include <string>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "number_plate_detector.hpp"

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
        "\tUsage: <image> [-t, --train; -h, --help]" << std::endl;
}

static const char* params = 
    "{ 1 | image |       | Path to target image }"
    "{ h | help  | false | Print help           }"
    "{ t | train | false | Train classifier     }";

int main(int argc, char* argv[]) {
	cv::CommandLineParser parser(argc, argv, params);
    std::string imagePath = parser.get<std::string>("image");
	bool train = parser.get<bool>("train");

    if (parser.get<bool>("help") || parser.get<bool>("train") && parser.get<bool>("image"))  {
        printHelp();
        parser.printParams();
        return 0;
    }
    
    if (train == true) {
        try {
            TNumberPlateDetector detector;
            detector.train();
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }

        return 0;
    }

    if (imagePath.empty() == false) {
        cv::Mat image = cv::imread(imagePath);
        CV_Assert(image.empty() == false);

        TNumberPlateDetector detector;
        
        std::cout << "Image: " << imagePath << std::endl;
        std::cout << "Detected plate number: " << detector.getNumber(image) << std::endl;
        return 0;
    }

	return 0;
}
