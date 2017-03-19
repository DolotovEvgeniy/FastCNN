// Copyright 2016 Dolotov Evgeniy

#ifndef BOUNDING_BOX_H_
#define BOUNDING_BOX_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class BoundingBox {
 public:
    double confidence;
    cv::Rect rect;

    bool operator<(BoundingBox object) {
        return confidence < object.confidence;
    }
};

#endif  // BOUNDING_BOX_H_
