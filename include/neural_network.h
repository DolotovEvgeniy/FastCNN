// Copyright 2016 Dolotov Evgeniy

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

#include <vector>
#include <string>

#include "heatmap.h"

class NeuralNetwork
{
public:
    NeuralNetwork() {}
    NeuralNetwork(std::string configFile, std::string trainedModel);
    void processImage(const cv::Mat& img, Heatmap& map);
    cv::Size inputLayerSize();
    cv::Size outputLayerSize();
    void resizeInputLayer(cv::Size size);
private:
    caffe::shared_ptr<caffe::Net<float> > net;

    void fillNeuralNetInput(const cv::Mat& img);
    void getNeuralNetOutput(Heatmap& map);
    void calculate();
};

#endif // NEURAL_NETWORK_H
