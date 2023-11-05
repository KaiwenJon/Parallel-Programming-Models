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

// Available versions: --sequential --openmp
// for example: ./houghTest --openmp ../data/coins.jpg
// Result will be saved to ../result/detection.jpg
```

![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/e78f5b63-1445-4ef4-b6a9-dca97c7f82c0)
