# SIFT Detector
Sequential version implemented.

## Algorithm
Using DoG (Difference of Gaussian) to build a scale space. From there, find interest points using NMS (Non-maximum suppression). Draw each SIFT interest point on the image with its feature scale.


## To run the code
```
mkdir build && cd build
cmake ..
make
./opencvTest ../data/buttefly.jpg

// Result will be saved to ../result.detection.jpg
```
