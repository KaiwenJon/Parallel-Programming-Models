#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include <pthread.h>
using namespace std;
#define SEQUENTIAL_VERSION 1
#define OPENMP_VERSION 2
#define PTHREAD_VERSION 3

static int version = SEQUENTIAL_VERSION;

pthread_mutex_t resultMutex;
struct Point{
    int cx;
    int cy;
    float radius;
    Point(int cx, int cy, float radius) : cx(cx), cy(cy), radius(radius){};
};
class SIFT_Detector;
struct ThreadData {
    int startRow;
    int endRow;
    vector<Point>* result;
    const vector<cv::Mat>* scaleSpace;
    const vector<float>* sigmas;
    SIFT_Detector* detector;
    int kernel_width;
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
        vector<Point> interestPoints;
        if(version == SEQUENTIAL_VERSION){
           interestPoints = extractInterestPoints_SEQ(scaleSpace);
        }
        else if(version == OPENMP_VERSION){
            interestPoints = extractInterestPoints_OPENMP(scaleSpace);
        }
        else if(version == PTHREAD_VERSION){
            interestPoints = extractInterestPoints_PTHREAD(scaleSpace);
        }
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

    static bool isMaximumNeighbors(int i, int j, int k, int kernel_width, const vector<cv::Mat>& scaleSpace){
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

    vector<Point> extractInterestPoints_SEQ(const vector<cv::Mat>& scaleSpace){
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



    vector<Point> extractInterestPoints_OPENMP(const vector<cv::Mat>& scaleSpace){
        vector<Point> result;
        int kernel_width=5;
        int rows = scaleSpace[0].rows;
        int cols = scaleSpace[0].cols;
        int depth = scaleSpace.size();
        // for each point in scaleSpace (h,w,layer), we check if it's the maxmimum of its neighbors
        // neighbors: 3x3xlayer
        auto start = chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                for(int k=0; k<depth; k++){
                    // cout << i << " " << j << " " << k << endl;
                    if(isMaximumNeighbors(i, j, k, kernel_width, scaleSpace)){
                        float radius = sigmas[k] * 1.414;
                        #pragma omp critical
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

    static void* extractInterestPoints_Thread(void* arg) {
        ThreadData* data = (ThreadData*)arg;
        int depth = data->scaleSpace->size();
        for(int i=data->startRow; i<data->endRow; i++){
            for(int j=0; j<(*data->scaleSpace)[0].cols; j++){
                for(int k=0; k<depth; k++){
                    if(data->detector->isMaximumNeighbors(i, j, k, data->kernel_width, *data->scaleSpace)){
                        float radius = (*data->sigmas)[k] * 1.414;
                        pthread_mutex_lock(&resultMutex);  // Lock the mutex
                        data->result->push_back({i, j, radius});
                        pthread_mutex_unlock(&resultMutex);  // Unlock the mutex
                    }
                }
            }
        }
        return NULL;
    }
    vector<Point> extractInterestPoints_PTHREAD(const vector<cv::Mat>& scaleSpace){
        vector<Point> result;
        int kernel_width = 5;
        int rows = scaleSpace[0].rows;
        int numThreads = 4;  // use 4 threads
        pthread_t threads[numThreads];
        ThreadData data[numThreads];

        // Initialize the mutex
        pthread_mutex_init(&resultMutex, NULL);

        int rowsPerThread = rows / numThreads;

        for (int i = 0; i < numThreads; i++) {
            data[i].startRow = i * rowsPerThread;
            data[i].endRow = (i == numThreads - 1) ? rows : (i + 1) * rowsPerThread;
            data[i].result = &result;
            data[i].scaleSpace = &scaleSpace;
            data[i].sigmas = &sigmas;
            data[i].detector = this;
            data[i].kernel_width = kernel_width;
            pthread_create(&threads[i], NULL, extractInterestPoints_Thread, (void*)&data[i]);
        }

        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < numThreads; i++) {
            pthread_join(threads[i], NULL);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken by feature extraction: " << duration.count() << " milliseconds" << std::endl;
        

        // Destroy the mutex
        pthread_mutex_destroy(&resultMutex);

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
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <version> <input_image_path>" << endl;
        cerr << "Available versions: --sequential, --openmp" << endl;
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
    
    int width = rgb_image.cols;
    int height = rgb_image.rows;

    cv::Mat grayscaleImage;
    cv::cvtColor(rgb_image, grayscaleImage, cv::COLOR_BGR2GRAY);

    cv::Mat floatImage;
    grayscaleImage.convertTo(floatImage, CV_32F);
    float sigma = 2.0f;
    SIFT_Detector detector(sigma);
    vector<Point> interestPoints = detector.detect(floatImage);
    cout << "Number of Detected Features: " << interestPoints.size() << endl;
    detector.visualizePoints(rgb_image, interestPoints);


    return 0;
}
