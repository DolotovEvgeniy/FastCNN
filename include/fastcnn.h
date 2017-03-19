// Copyright 2016 Dolotov Evgeniy

#ifndef UNITBOX_H
#define UNITBOX_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include <cmath>

#include "neural_network.h"
#include "heatmap.h"

const int DETECTOR_SIZE = 32;
const float SCALE = sqrt(2);
const int LEVEL_COUNT = 10;
class FastCNNDetector {
public:
    FastCNNDetector(std::string netConfiguration,
                    std::string pretrainNetwork);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects);
private:
    NeuralNetwork net;
};

#endif // UNITBOX_H
