# Hough Circle Transform
Sequential version implemented.

## Algorithm
Given an image, build the parameter space and do the voting to extract circle features.


## To run the code
```
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4 // if using --openmp
./houghTest <version> <input_image_path>

// Available versions: --sequential
// for example: ./houghTest --openmp ../data/coins.jpg
// Result will be saved to ../result/detection.jpg
```

