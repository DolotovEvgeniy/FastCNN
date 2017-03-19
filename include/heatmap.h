// Copyright 2016 Dolotov Evgeniy

#ifndef HEATMAP_H
#define HEATMAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

class Heatmap {
public:
    Heatmap() = default;
    Heatmap(int width, int height);
    Heatmap(int width, int height, const float* data);
    Heatmap(const Heatmap& map);
    Heatmap& operator=(const Heatmap& map);
    cv::Size size() const;
    float& at(int x, int y);
    float at(int x, int y) const;
    void show(std::string windowName) const;
private:
    cv::Mat map;
};

#endif // HEATMAP_H
