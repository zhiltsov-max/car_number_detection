#ifndef NUMBER_PLATE_DETECTOR_H
#define NUMBER_PLATE_DETECTOR_H

#include <string>
#include <iostream>
#include "opencv2\core\core.hpp"


class TNumberPlateDetector {
public:
    class Number : public std::vector<std::string> {
    public:
        operator std::string() const;

        friend std::ostream& operator <<(std::ostream& os, const Number& number) {
            return os << static_cast<std::string>(number);
        }
    };

    Number getNumber(const cv::Mat& frame);
    void train();
private:
    class Recognizer;
};

#endif // NUMBER_PLATE_DETECTOR_H