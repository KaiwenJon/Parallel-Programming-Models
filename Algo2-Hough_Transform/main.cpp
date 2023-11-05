#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include <pthread.h>
#include <mpi.h>

using namespace std;


typedef enum {
    SEQUENTIAL_VERSION,
    OPENMP_VERSION,
    PTHREAD_VERSION,
    mpi_VERSION
} Version;

struct Point{
    int cx;
    int cy;
    int radius;
    Point(): cx(0), cy(0), radius(0) {};
    Point(int cx, int cy, int radius) : cx(cx), cy(cy), radius(radius){};
};

class Hough_Circle{
public:
    Hough_Circle(Version version): version(version) {}

    vector<Point> detect(const cv::Mat& inputImage){
        int height = inputImage.rows;
        int width = inputImage.cols;

        vector<int> radius_candidate = {20, 25, 28, 30, 85};
        
        vector<cv::Mat> parameterSpace;
        for (const int& r: radius_candidate) {
            cv::Mat layer(height, width, CV_32S, cv::Scalar(0));
            parameterSpace.push_back(layer);
        }
        cv::Mat edges;
        cv::Canny(inputImage, edges, 255, 255);
        // cv::imshow("Hollow Circles", edges);
        // cv::waitKey(0);

        cout << "Voting in parameter space..." << endl;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (edges.at<uchar>(y, x) > 0) { // If it's an edge point
                    for (int i=0; i<radius_candidate.size(); i++) {
                        int r = radius_candidate[i];
                        for (int theta = 0; theta < 360; theta++) {
                            int a = x - r * cos(theta * CV_PI / 180);
                            int b = y - r * sin(theta * CV_PI / 180);
                            if (a >= 0 && a < width && b >= 0 && b < height) {
                                parameterSpace[i].at<int>(b, a)++;
                            }
                        }
                    }
                }
            }
        }

        cout << "Finding maximum in parameter space..." << endl;
        // Find circle candidates in the accumulator matrix by thresholding.
        int threshold = 150; // Adjust this threshold as needed.
        vector<Point> circles; // (x, y, radius)
        for (int i=0; i<radius_candidate.size(); i++) {
            int r = radius_candidate[i];
            // double minVal, maxVal;
            // cv::Point minLoc, maxLoc;
            // cv::minMaxLoc(parameterSpace[i], &minVal, &maxVal, &minLoc, &maxLoc);
            // std::cout << "Min Value: " << minVal << std::endl;
            // std::cout << "Max Value: " << maxVal << std::endl;
            // cv::Mat displayImage;
            // cv::normalize(parameterSpace[i], displayImage, 0, 255, cv::NORM_MINMAX, CV_8U);
            // cv::imshow("Grayscale Image" + to_string(r), displayImage);
            // cv::waitKey(0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (parameterSpace[i].at<int>(y, x) >= threshold && isMaximumNeighbor(parameterSpace[i], y, x)) {
                        circles.push_back({x, y, r});
                    }
                }
            }
        }

        return circles;
    }

    bool isMaximumNeighbor(const cv::Mat& layer, int y, int x){
        int r = 10;
        for(int i=y-r; i<=y+r; i++){
            for(int j=x-r; j<=x+r; j++){
                if(i >= 0 && i <= layer.rows && j >=0 && j<=layer.cols && layer.at<int>(i, j) > layer.at<int>(y, x)){
                    return false;
                }
            }
        }
        return true;
    }


    void visualizeCircles(cv::Mat& rgb_image, vector<Point>& circles){
        cout << "Found " << circles.size() << " circles. " << endl;
        for(const Point& circle: circles){
            cv::circle(rgb_image, cv::Point(circle.cx, circle.cy), static_cast<int>(circle.radius), cv::Scalar(0, 0, 255), 2);  // Red circle, thickness 2 
            cout << "cx: " << circle.cx << ", cy: " << circle.cy << ", r: " << circle.radius << endl;
        }
        // cv::imshow("Hollow Circles", rgb_image);
        // cv::waitKey(0);
        cout << "Result saved." << endl;
        cv::imwrite("../result/detection.jpg", rgb_image);
    }

private:
    Version version = SEQUENTIAL_VERSION;
};

int main(int argc, char** argv) {
    Version version = SEQUENTIAL_VERSION;
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <version> <input_image_path>" << endl;
        cerr << "Available versions: --sequential" << endl;
        return 1;
    }
    if(strcmp(argv[1], "--openmp") == 0){
        version = OPENMP_VERSION;
        cout << "Running openmp version." << endl;
    }
    else if(strcmp(argv[1], "--pthread") == 0){
        version = PTHREAD_VERSION;
        cout << "Running pthread version." << endl;
    }
    else if(strcmp(argv[1], "--mpi") == 0){
        version = mpi_VERSION;
        cout << "Running mpi version." << endl;
    }
    else{
        version = SEQUENTIAL_VERSION;
        cout << "Running sequential version." << endl;
    }

    const char* inputImagePath = argv[2];
    cv::Mat rgb_image = cv::imread(inputImagePath);


    if (rgb_image.empty()) {
        cerr << "Error: Unable to load the image." << endl;
        return 1;
    }
    double scale = 1;
    cv::Size newSize(static_cast<int>(rgb_image.cols * scale), static_cast<int>(rgb_image.rows * scale));
    cv::Mat resizedImage;
    cv::resize(rgb_image, resizedImage, newSize);
    int width = resizedImage.cols;
    int height = resizedImage.rows;

    cv::Mat grayscaleImage;
    cv::cvtColor(resizedImage, grayscaleImage, cv::COLOR_BGR2GRAY);


    Hough_Circle detector(version);
    auto start = chrono::high_resolution_clock::now();
    vector<Point> circles = detector.detect(grayscaleImage);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

    cout << "Time taken by the whole algo: " << duration.count() << " milliseconds" << std::endl;
        
    detector.visualizeCircles(rgb_image, circles);
    
    return 0;
}
