#include "fastcnn.h"
#include "nms.h"
#include "bounding_box.h"
using namespace cv;
using namespace std;


FastCNNDetector::FastCNNDetector(string netConfiguration,
                                 string pretrainNetwork)
                                 : net(netConfiguration, pretrainNetwork) {
}

void FastCNNDetector::detect(const Mat& image,
                             vector<Rect>& objects) {
    CV_Assert(image.channels() == 3);
    int initialWidth;
    int initialHeight;
    if ( image.cols > image.rows ) {
        initialWidth = DETECTOR_SIZE*image.cols/float(image.rows);
        initialHeight = DETECTOR_SIZE;
    } else {
        initialWidth = DETECTOR_SIZE;
        initialHeight = DETECTOR_SIZE*image.rows/float(image.cols);
    }
    vector<BoundingBox> detectedObjects;
    for (int i = 0; i < LEVEL_COUNT; i++) {
        Mat resizedImage;
        Size size(initialWidth*pow(SCALE, i), initialHeight*pow(SCALE, i));
        cout << "LEVEL" << i << ": "<< size << endl;
        resize(image, resizedImage, size);
        Heatmap map;
        net.processImage(resizedImage, map);
        float levelScale = image.cols/float(resizedImage.cols);
        int count = 0;
        for (int x = 0; x < map.size().width; x++) {
            for ( int y = 0; y < map.size().height; y++) {
                if(map.at(x, y) > 0.6) {
                    Rect rect(Point(2*x*levelScale, 2*y*levelScale),
                              Size(DETECTOR_SIZE*levelScale, DETECTOR_SIZE*levelScale));
                    BoundingBox box;
                    box.rect = rect;
                    box.confidence = map.at(x, y);
                    detectedObjects.push_back(box);
                    count++;
                }
            }
        }
    }
    NMSmax nms;
    nms.processBondingBox(detectedObjects, 0.2, 0.7);
    for (int i = 0; i < detectedObjects.size(); i++) {
        objects.push_back(detectedObjects[i].rect);
        cout << detectedObjects[i].rect << endl;
    }
}
