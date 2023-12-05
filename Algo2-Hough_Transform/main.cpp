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

pthread_mutex_t mutexlock;

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

struct ThreadData {
    int startRow, endRow;
    cv::Mat* edges;
    vector<int> radius_candidate;
    vector<cv::Mat>* parameterSpace;
};

void* threadFunction(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    for (int r_idx = 0; r_idx < data->radius_candidate.size(); r_idx++) {
        int r = data->radius_candidate[r_idx];
        for (int y = data->startRow; y < data->endRow; y++) {
            for (int x = 0; x < data->edges->cols; x++) {
                if (data->edges->at<uchar>(y, x) > 0) {
                    for (int theta = 0; theta < 360; theta++) {
                        int a = x - r * cos(theta * CV_PI / 180);
                        int b = y - r * sin(theta * CV_PI / 180);
                        if (a >= 0 && a < data->edges->cols && b >= 0 && b < data->edges->rows) {
                            int& count = data->parameterSpace->at(r_idx).at<int>(b, a);
                            __sync_fetch_and_add(&count, 1);
                        }
                    }
                }
            }
        }
    }

    pthread_exit(NULL);
}

struct MaxSearchData {
    int startRow, endRow;
    int cols;
    vector<cv::Mat>* parameterSpace;
    vector<int> radius_candidate;
    int threshold;
    vector<Point>* localCircles;
};

void* maxSearchThread(void* arg) {
    MaxSearchData* data = (MaxSearchData*)arg;

    for (int r_idx = 0; r_idx < data->radius_candidate.size(); r_idx++) {
        int r = data->radius_candidate[r_idx];
        for (int y = data->startRow; y < data->endRow; y++) {
            for (int x = 0; x < data->cols; x++) {
                if (data->parameterSpace->at(r_idx).at<int>(y, x) >= data->threshold && isMaximumNeighbor(data->parameterSpace->at(r_idx), y, x)) {
                    pthread_mutex_lock(&mutexlock);
                    data->localCircles->push_back({x, y, r});
                    pthread_mutex_unlock(&mutexlock);
                }
            }
        }
    }
    pthread_exit(NULL);
}


class Hough_Circle{
public:
    Hough_Circle(Version version): version(version) {}

    vector<Point> _detect_Seq(const cv::Mat& inputImage){
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

        auto start = chrono::high_resolution_clock::now();
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
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to build param space: " << duration.count() << " milliseconds" << std::endl;
        
        cout << "Finding maximum in parameter space..." << endl;

        start = chrono::high_resolution_clock::now();
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
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to find circles in param space: " << duration.count() << " milliseconds" << std::endl;
       

        return circles;
    }

    vector<Point> _detect_Openmp(const cv::Mat& inputImage){
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

        cout << "Voting in parameter space..." << endl;

        auto start = chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(3)
        for(int r_idx=0; r_idx<radius_candidate.size(); r_idx++){
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = radius_candidate[r_idx];
                    if (edges.at<uchar>(y, x) > 0) { // If it's an edge point
                        for (int theta = 0; theta < 360; theta++) {
                            int a = x - r * cos(theta * CV_PI / 180);
                            int b = y - r * sin(theta * CV_PI / 180);
                            if (a >= 0 && a < width && b >= 0 && b < height) {
                                #pragma omp atomic
                                parameterSpace[r_idx].at<int>(b, a)++;
                            }
                        }
                        
                    }
                }
            }
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to build param space: " << duration.count() << " milliseconds" << std::endl;
        
        cout << "Finding maximum in parameter space... with openmp" << endl;
        start = chrono::high_resolution_clock::now();
        // Find circle candidates in the accumulator matrix by thresholding.
        int threshold = 150; // Adjust this threshold as needed.
        vector<Point> circles; // (x, y, radius)

        #pragma omp parallel for collapse(3)
        for (int i=0; i<radius_candidate.size(); i++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = radius_candidate[i];
                    if (parameterSpace[i].at<int>(y, x) >= threshold && isMaximumNeighbor(parameterSpace[i], y, x)) {
                        #pragma omp critical
                        circles.push_back({x, y, r});
                    }
                }
            }
        }
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to find circles in param space: " << duration.count() << " milliseconds" << std::endl;
        
        return circles;
    }

    vector<Point> _detect_pthread(const cv::Mat& inputImage){
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


        // First phase: build the parameter space.
        cout << "Voting in parameter space... with pthread" << endl;

        auto start = chrono::high_resolution_clock::now();
        int numThreads = 4;
        pthread_t threads[numThreads];
        ThreadData threadData[numThreads];

        int rowsPerThread = height / numThreads;
        for(int i=0; i<numThreads; i++){
            threadData[i].startRow = i * rowsPerThread;
            threadData[i].endRow = (i == numThreads - 1) ? height : (i+1) * rowsPerThread;
            threadData[i].edges = &edges;
            threadData[i].radius_candidate = radius_candidate;
            threadData[i].parameterSpace = &parameterSpace;

            pthread_create(&threads[i], NULL, threadFunction, (void*)&threadData[i]);
        }

        for(int i=0; i<numThreads; i++){
            pthread_join(threads[i], NULL);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to build param space: " << duration.count() << " milliseconds" << std::endl;
        
        // up to this point, parameterSpace is ready to be searched
        // Second phase: detecting maximums
        cout << "Finding maximum in parameter space... with pthread" << endl;
        start = chrono::high_resolution_clock::now();
        // Find circle candidates in the accumulator matrix by thresholding.
        int threshold = 150; // Adjust this threshold as needed.
        vector<Point> globalCircles;
        pthread_t maxThreads[numThreads];
        MaxSearchData maxThreadData[numThreads];
        vector<vector<Point>> localCircles(numThreads);
        for (int i = 0; i < numThreads; i++) {
            maxThreadData[i].startRow = i * rowsPerThread;
            maxThreadData[i].endRow = (i == numThreads - 1) ? height : (i+1) * rowsPerThread;
            maxThreadData[i].cols = width;
            maxThreadData[i].parameterSpace = &parameterSpace;
            maxThreadData[i].radius_candidate = radius_candidate;
            maxThreadData[i].threshold = threshold;
            maxThreadData[i].localCircles = &localCircles[i];

            pthread_create(&maxThreads[i], NULL, maxSearchThread, (void*)&maxThreadData[i]);
        }

        // Wait for threads to finish
        for (int i = 0; i < numThreads; i++) {
            pthread_join(threads[i], NULL);
            globalCircles.insert(globalCircles.end(), localCircles[i].begin(), localCircles[i].end());
        }
        end = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to find circles in param space: " << duration.count() << " milliseconds" << std::endl;
        
        return globalCircles;
    }

    vector<Point> _detect_mpi(const cv::Mat& inputImage){
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int height = inputImage.rows;
        int width = inputImage.cols;
        int rows_per_process = height / size;
        int startRow = rank * rows_per_process;
        int endRow = (rank == size - 1)? height : (rank+1) * rows_per_process;

        vector<int> radius_candidate = {20, 25, 28, 30, 85};
        vector<cv::Mat> local_parameterSpace;
        for (const int& r: radius_candidate) {
            cv::Mat layer(height, width, CV_32S, cv::Scalar(0));
            local_parameterSpace.push_back(layer);
        }

        cv::Mat edges;
        cv::Canny(inputImage, edges, 255, 255);

        cout << "Voting in parameter space... with pthread, rank: " << rank  << endl;

        auto start = chrono::high_resolution_clock::now();
        for (int y = startRow; y < endRow; y++) {
            for (int x = 0; x < width; x++) {
                if (edges.at<uchar>(y, x) > 0) { // If it's an edge point
                    for (int i=0; i<radius_candidate.size(); i++) {
                        int r = radius_candidate[i];
                        for (int theta = 0; theta < 360; theta++) {
                            int a = x - r * cos(theta * CV_PI / 180);
                            int b = y - r * sin(theta * CV_PI / 180);
                            if (a >= 0 && a < width && b >= 0 && b < height) {
                                local_parameterSpace[i].at<int>(b, a)++;
                            }
                        }
                    }
                }
            }
        }
        vector<cv::Mat> global_parameterSpace;
        if(rank == 0){
            for (const int& r: radius_candidate) {
                cv::Mat layer(height, width, CV_32S, cv::Scalar(0));
                global_parameterSpace.push_back(layer);
            }
        }
        // MPI_Gather(local_parameterSpace.data(), height * width * radius_candidate.size(), MPI_INT, 
        //     global_parameterSpace.data(), height * width * radius_candidate.size(), MPI_INT, 0, MPI_COMM_WORLD);
        for (auto& image : local_parameterSpace) {
            if (!image.isContinuous()) {
                cout << "local not conti!!\n";
                image = image.clone(); // Create a contiguous copy of the image data
            }
        }
        for (auto& image : global_parameterSpace) {
            if (!image.isContinuous()) {
                cout << "global not conti!!\n";
                image = image.clone(); // Create a contiguous copy of the image data
            }
        }
        
        MPI_Gather(
            local_parameterSpace.data(), // Local data pointer
            local_parameterSpace.size(), // Number of local images
            MPI_BYTE, // Data type (cv::Mat is a complex data structure)
            global_parameterSpace.data(), // Global data pointer
            local_parameterSpace.size(), // Number of local images (same as send count)
            MPI_BYTE, // Data type (cv::Mat is a complex data structure)
            0, // Root process (rank 0)
            MPI_COMM_WORLD
        );

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end-start);

        cout << "Time taken to build param space: " << duration.count() << " milliseconds, rank: " << rank << std::endl;
        
        vector<Point> circles;
        int threshold = 150;
        if(rank == 0){
            for (int i=0; i<radius_candidate.size(); i++) {
                cout << global_parameterSpace[i].size() << endl;
                int r = radius_candidate[i];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (global_parameterSpace[i].at<int>(y, x) >= threshold && isMaximumNeighbor(global_parameterSpace[i], y, x)) {
                            circles.push_back({x, y, r});
                        }
                    }
                }
            }
        }

        return circles;
    }

    vector<Point> detect(const cv::Mat& inputImage){
        if(version == SEQUENTIAL_VERSION){
            return _detect_Seq(inputImage);
        }
        else if (version == OPENMP_VERSION){
            return _detect_Openmp(inputImage);
        }
        else if (version == PTHREAD_VERSION){
            return _detect_pthread(inputImage);
        }
        else if (version == mpi_VERSION){
            return _detect_mpi(inputImage);
        }
        else{
            return _detect_Seq(inputImage);
        }
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
        if(version != mpi_VERSION || rank == 0){
            cout << "Result saved." << endl;
            cv::imwrite("../result/detection.jpg", rgb_image);
        }
    }

private:
    Version version = SEQUENTIAL_VERSION;
    int rank, size;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    pthread_mutex_init(&mutexlock, NULL);
    Version version = SEQUENTIAL_VERSION;
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <version> <input_image_path>" << endl;
        cerr << "Available versions: --sequential --openmp  --pthread --mpi" << endl;
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
    
    pthread_mutex_destroy(&mutexlock);

    MPI_Finalize();
    return 0;
}
