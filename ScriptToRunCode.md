# ParallelizingMandelbrot

Note that you need to have the MPICC library installed:

In case you do not have it on your system:

For Kali Linux:

sudo apt-get -y install mpich

To compile the code:

mpicc -o dynamic dynamic.c

To run the code:

mpirun -np 4 ./dynamic
