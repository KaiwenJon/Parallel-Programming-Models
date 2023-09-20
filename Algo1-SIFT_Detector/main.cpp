#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp> // Include OpenCV headers
using namespace std;

struct Point{
    int cx;
    int cy;
    float radius;
    Point(int cx, int cy, float radius) : cx(cx), cy(cy), radius(radius){};
};

class SIFT_Detector{
public:
    SIFT_Detector(float sigma){
        sigmas = {sigma};
        for(int i=0; i<10; i++){
            sigmas.push_back(sigmas.back() * 1.5);
        }
    };

    vector<Point> detect(const cv::Mat& inputImage){
        vector<cv::Mat> scaleSpace = buildScaleSpace(inputImage);
        vector<Point> interestPoints = extractInterestPoints(scaleSpace);
        return interestPoints; 
    }

    vector<cv::Mat> buildScaleSpace(const cv::Mat& inputImage){
        vector<cv::Mat> result;
        vector<cv::Mat> blurredImages;
        for(int sigma: sigmas){
            cv::Mat blurred;
            int kernel_size = round(sigma*6);
            if(kernel_size % 2 == 0){
                kernel_size++;
            }
            cv::GaussianBlur(inputImage, blurred, cv::Size(kernel_size, kernel_size), sigma);
            blurredImages.push_back(blurred);
        }
        for(int i=1; i<blurredImages.size(); i++){
            cv::Mat diff = blurredImages[i] - blurredImages[i-1];
            cv::Mat squared = diff.mul(diff);
            result.push_back(squared);
        }

        // Save the sigma space images
        for (size_t i = 0; i < result.size(); ++i) {
            cv::Mat norm;
            cv::normalize(result[i], norm, 0, 255, cv::NORM_MINMAX);
            norm.convertTo(norm, CV_8U);
            cv::imwrite("../test/sigma_space_" + std::to_string(sigmas[i]) + ".jpg", norm);
        }

        return result;
    }

    bool isMaximumNeighbors(int i, int j, int k, int kernel_width, const vector<cv::Mat>& scaleSpace){
        int rows = scaleSpace[0].rows;
        int cols = scaleSpace[0].cols;
        int depth = scaleSpace.size();
        float central_value = scaleSpace[k].at<float>(i, j);
        for(int r=i-kernel_width/2; r<=i+kernel_width/2; r++){
            for(int c=j-kernel_width/2; c<=j+kernel_width/2; c++){
                for(int d=0; d<depth; d++){
                    if(r < 0 || c < 0 || r > rows-1 || c > cols-1){
                        continue;
                    }
                    float neighbor_value = scaleSpace[d].at<float>(r, c);
                    if(central_value - neighbor_value < 0){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    vector<Point> extractInterestPoints(const vector<cv::Mat>& scaleSpace){
        vector<Point> result;
        int kernel_width=5;
        int rows = scaleSpace[0].rows;
        int cols = scaleSpace[0].cols;
        int depth = scaleSpace.size();
        // for each point in scaleSpace (h,w,layer), we check if it's the maxmimum of its neighbors
        // neighbors: 3x3xlayer
        auto start = chrono::high_resolution_clock::now();
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                for(int k=0; k<depth; k++){
                    // cout << i << " " << j << " " << k << endl;
                    if(isMaximumNeighbors(i, j, k, kernel_width, scaleSpace)){
                        float radius = sigmas[k] * 1.414;
                        result.push_back({i, j, radius});
                        // cout << sigmas[k] << " ";
                    }
                }
            }
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken by feature extraction: " << duration.count() << " milliseconds" << std::endl;
        return result;
    }

    void visualizePoints(const cv::Mat& rgb_image, const vector<Point>& interestPoints){
        for(const Point& circle: interestPoints){
           cv::circle(rgb_image, cv::Point(circle.cy, circle.cx), static_cast<int>(circle.radius), cv::Scalar(0, 0, 255), 2);  // Red circle, thickness 2 
        }
        // cv::imshow("Hollow Circles", rgb_image);
        // cv::waitKey(0);
        cout << "Result saved." << endl;
        cv::imwrite("../result/detection.jpg", rgb_image);
    }
private:
    vector<float> sigmas;
};


int main(int argc, char** argv) {
    
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_image_path>" << endl;
        return 1;
    }
    const char* inputImagePath = argv[1];
    cv::Mat rgb_image = cv::imread(inputImagePath);

    if (rgb_image.empty()) {
        cerr << "Error: Unable to load the image." << endl;
        return 1;
    }
    
    int width = rgb_image.cols;
    int height = rgb_image.rows;

    cv::Mat grayscaleImage(height, width, CV_8U);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3b rgbPixel = rgb_image.at<cv::Vec3b>(i, j);

            int r = rgbPixel[2];  
            int g = rgbPixel[1];  
            int b = rgbPixel[0];  
            int grayscaleValue = (r + g + b) / 3;

            grayscaleImage.at<uchar>(i, j) = grayscaleValue;
        }
    }

    cv::Mat floatImage;
    grayscaleImage.convertTo(floatImage, CV_32F);
    float sigma = 2.0f;
    SIFT_Detector detector(sigma);
    vector<Point> interestPoints = detector.detect(floatImage);
    cout << "Number of Detected Features: " << interestPoints.size() << endl;
    detector.visualizePoints(rgb_image, interestPoints);


    return 0;
}
