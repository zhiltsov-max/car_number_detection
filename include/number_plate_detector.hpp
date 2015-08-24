#ifndef NUMBER_PLATE_DETECTOR_H
#define NUMBER_PLATE_DETECTOR_H

#include <string>
#include "opencv2\core\core.hpp"

class TNumberPlateDetector {
public:
    typedef std::string Number;

    Number getNumber(const cv::Mat& frame);

private:
};

#endif // NUMBER_PLATE_DETECTOR_H