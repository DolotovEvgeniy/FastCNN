// Copyright 2016 Dolotov Evgeniy

#include "neural_network.h"
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;
using namespace caffe;

NeuralNetwork::NeuralNetwork(string configFile, string trainedModel) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    net.reset(new Net<float>(configFile, caffe::TEST));
    net->CopyTrainedLayersFrom(trainedModel);
    Blob<float>* input_layer = net->input_blobs()[0];
    assert(input_layer->width() == input_layer->height());
}

void NeuralNetwork::processImage(const Mat &img, Heatmap& map) {
    fillNeuralNetInput(img);
    calculate();
    getNeuralNetOutput(map);
}

void NeuralNetwork::fillNeuralNetInput(const Mat &img) {
    Blob<float>* input_layer = net->input_blobs()[0];
    int width = img.cols;
    int height = img.rows;
    input_layer->Reshape(1, input_layer->channels(), height, width);
    net->Reshape();

    vector<Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    split(img_float, input_channels);
}

void NeuralNetwork::getNeuralNetOutput(Heatmap& map) {
     Blob<float>* output_layer = net->output_blobs()[0];

     int width  = output_layer->width();
     int height = output_layer->height();
     map = Heatmap(width, height, output_layer->cpu_data()+width*height);
}

void NeuralNetwork::calculate() {
    net->ForwardPrefilled();
}

Size NeuralNetwork::inputLayerSize()  {
    Blob<float>* input_layer = net->input_blobs()[0];
    return Size(input_layer->width(), input_layer->height());
}

Size NeuralNetwork::outputLayerSize() {
    Blob<float>* output_layer = net->output_blobs()[0];
    return Size(output_layer->width(), output_layer->height());
}

void NeuralNetwork::resizeInputLayer(Size size) {
    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, input_layer->channels(), size.height, size.width);
    net->Reshape();
}
