# Hough Circle Transform
Sequential, openmp, pthread, and mpi version implemented.

## Algorithm
Given an image, build the parameter space and do the voting to extract circle features.


## To run the code
```
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4 // if using --openmp
./houghTest <version> <input_image_path>

// Available versions: --sequential --openmp --pthread --mpi
// for example: ./houghTest --openmp ../data/coins.jpg
// Result will be saved to ../result/detection.jpg
// If using mpi, run this command:
mpiexec -n <number of threads> --host localhost:8 ./houghTest --mpi ../data/coins.jpg
```

![image](https://github.com/KaiwenJon/Parallel-Programming-Models/assets/70893513/e78f5b63-1445-4ef4-b6a9-dca97c7f82c0)
