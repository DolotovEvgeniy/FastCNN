#include "heatmap.h"
#include <iostream>

using namespace std;
using namespace cv;

Heatmap::Heatmap(int width, int height) {
    map = Mat::zeros(height, width, CV_8UC1);
}

Heatmap::Heatmap(int width, int height, const float* data) {
    float *tmpData = new float[width*height];
    for(int i = 0; i<width*height; i++) {
        tmpData[i] = data[i];
    }
    map = Mat(height, width, CV_32FC1, tmpData);
}


Heatmap::Heatmap(const Heatmap& map) {
    map.map.copyTo(this->map);
}

Heatmap& Heatmap::operator=(const Heatmap& map) {
    map.map.copyTo(this->map);
    return *this;
}

float& Heatmap::at(int x, int y) {
    return map.at<float>(Point(x, y));
}

float Heatmap::at(int x, int y) const {
    return map.at<float>(Point(x, y));
}

void Heatmap::show(std::string windowName) const {
    Mat picture;
    cv::normalize(map, picture, 0, 255, cv::NORM_MINMAX);
    cout << "Prob" << map.at<float>(Point(0, 0)) << endl;
    imshow(windowName, picture);

    waitKey(0);
}

Size Heatmap::size() const {
    return Size(map.cols, map.rows);
}
