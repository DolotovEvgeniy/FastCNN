// Copyright 2016 Dolotov Evgeniy

#include "../include/nms.h"
#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

double IOU(const Rect& rect1, const Rect& rect2) {
    Rect unionRect = rect1 & rect2;

    return unionRect.area()/(double)(rect1.area()+rect2.area()-unionRect.area());
}

Rect weightedAvg_rect(const vector<BoundingBox> &rectangles)
{
    CV_Assert(rectangles.size() > 0);

    Rect resultRect;

    double sumOfX = 0, sumOfY = 0, sumOfWidth = 0, sumOfHeight = 0;
    double weight = 0;
    for (size_t i = 0; i < rectangles.size(); i++) {
        weight += rectangles[i].confidence;
    }
    for (size_t i = 0; i < rectangles.size(); i++) {
        Rect rectangle = rectangles[i].rect;
        double c = rectangles[i].confidence;
        double d=c/weight;
        sumOfX+=rectangle.x*d;
        sumOfY+=rectangle.y*d;
        sumOfWidth+=rectangle.width*d;
        sumOfHeight+=rectangle.height*d;
    }
    resultRect.x = sumOfX;
    resultRect.y = sumOfY;
    resultRect.width = sumOfWidth;
    resultRect.height = sumOfHeight;
    return resultRect;
}

Rect avg_rect(const vector<Rect>& rectangles) {
    CV_Assert(rectangles.size() > 0);

    Rect resultRect;

    double sumOfX = 0, sumOfY = 0, sumOfWidth = 0, sumOfHeight = 0;
    for (size_t i = 0; i < rectangles.size(); i++) {
        sumOfX+=rectangles[i].x;
        sumOfY+=rectangles[i].y;
        sumOfWidth+=rectangles[i].width;
        sumOfHeight+=rectangles[i].height;
    }
    int n = rectangles.size();
    resultRect.x = sumOfX/n;
    resultRect.y = sumOfY/n;
    resultRect.width = sumOfWidth/n;
    resultRect.height = sumOfHeight/n;
    return resultRect;
}

Rect intersectRectangles(const vector<Rect>& rectangles) {
    Rect resultRectangle = rectangles[0];

    for (vector<Rect>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++) {
        resultRectangle = resultRectangle & (*rectangle);
    }

    return resultRectangle;
}

void NMS::divideIntoClusters(vector<BoundingBox>& objects, const double &box_threshold, vector<BoundingBoxCluster>& clusters) {
    while (!objects.empty()) {
        BoundingBox objectWithMaxConfidence = *max_element(objects.begin(), objects.end());
        BoundingBoxCluster cluster;
        vector<BoundingBox> newObjects;
        for (size_t i = 0; i < objects.size(); i++) {
            if (IOU(objectWithMaxConfidence.rect, objects[i].rect) <= box_threshold) {
                newObjects.push_back(objects[i]);
            } else {
                cluster.push_back(objects[i]);
            }
        }
        clusters.push_back(cluster);
        objects = newObjects;
    }
}

void NMS::processBondingBox(vector<BoundingBox> &objects, const double &box_threshold, const double &confidence_threshold) {
    vector<BoundingBox> detectedObjects;
    vector<BoundingBoxCluster> clusters;

    divideIntoClusters(objects, box_threshold, clusters);
    for (vector<BoundingBoxCluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); cluster++) {
        int boundBoxCount = cluster->size();
        cout << "Box in cluster:" << boundBoxCount << endl;
        BoundingBox box = mergeCluster(*cluster, confidence_threshold);
        detectedObjects.push_back(mergeCluster(*cluster, confidence_threshold));
    }
    objects = detectedObjects;
}

BoundingBox NMSmax::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    return *max_element(cluster.begin(), cluster.end());
}

BoundingBox NMSavg::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->rect);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = avg_rect(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;
    cout << "Confidence" << maxConfidenceInCluster << endl;
    return resultBoundingBox;
}

BoundingBox NMSweightedAvg::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<BoundingBox> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(*boundingBox);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = weightedAvg_rect(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;
cout << "Confidence" << maxConfidenceInCluster << endl;
    return resultBoundingBox;
}

BoundingBox NMSintersect::mergeCluster(BoundingBoxCluster &cluster, const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(), cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (vector<BoundingBox>::iterator boundingBox = cluster.begin(); boundingBox != cluster.end(); boundingBox++) {
        if (boundingBox->confidence > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->rect);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = intersectRectangles(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;

    return resultBoundingBox;
}
