#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;

    double z_real2, z_imag2, lengthsq;

    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq =  z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int num_trials = 10;
    double total_elapsed_time = 0;
    double total_comm_time = 0;

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int stop_signal = 0; 

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int trial = 0; trial < num_trials; trial++) {

        double start_time = MPI_Wtime();

        double comm_time = 0;
        int rows = 0;
        int count = 0;
        int buffer[10][WIDTH];
        struct complex c;
        int image[HEIGHT][WIDTH];
        MPI_Request request;
        double starttime, endtime;
        MPI_Status status; 

        if (rank == 0) {
            // Master process

            for (int i = 1; i < size; i++) {
                starttime = MPI_Wtime();
                MPI_Isend(&rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request); //send1
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                endtime = MPI_Wtime();
                comm_time += endtime - starttime;
                count += 10;
                rows += 10;
            }

            do {
                int source_tag = 0;
                starttime = MPI_Wtime();
                MPI_Irecv(image + rows, 10 * WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request); //recieve 2 
                MPI_Wait(&request, &status);
                endtime = MPI_Wtime();
                comm_time += endtime - starttime;
                count -= 10;
                if (rows < HEIGHT) {
                    starttime = MPI_Wtime();
                    if (rows == HEIGHT - 10){
                        MPI_Isend(&rows, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &request); //send1
                        MPI_Wait(&request, MPI_STATUS_IGNORE); 
                        stop_signal = 1;
                        // printf("%d count\n", count);
                    } else {
                        MPI_Isend(&rows, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request); //send1
                        MPI_Wait(&request, MPI_STATUS_IGNORE);
                        count += 10;
                        rows += 10;

                    }
                    endtime = MPI_Wtime();
                    comm_time += endtime - starttime;
                } else {
                    count -= 10;
                    starttime = MPI_Wtime();
                    MPI_Isend(&rows, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &request); // Signal termination
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                    endtime = MPI_Wtime();
                    comm_time += endtime - starttime;
                }

            } while (count > 0);

            save_pgm("dynamic.pgm", image);
        } else {
            // Slave processes
            int r, tag = 0; // Initialize tag
            while (stop_signal == 0) {
                starttime = MPI_Wtime();
                MPI_Irecv(&r, 1, MPI_INT, 0, MPI_ANY_TAG , MPI_COMM_WORLD, &request); //recieve 1
                MPI_Wait(&request, &status);
                printf("%d %d %d %d \n",status.MPI_TAG, r, rank, stop_signal);

                tag = status.MPI_TAG;
                endtime = MPI_Wtime();
                comm_time += endtime - starttime;
                if (tag == 1){                 
                  // printf("rank: %d breaks?\n" , rank);
                    break;
                }
                for (int x = 0; x < WIDTH; x++) {
                    for (int y = r; y < r + 10 && y<HEIGHT; y++) {
                        c.real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
                        c.imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
                        buffer[y - r][x] = cal_pixel(c); 
                    }
                }

                

                starttime = MPI_Wtime();
                MPI_Isend(buffer, 10 * WIDTH, MPI_INT, 0, rank, MPI_COMM_WORLD, &request); //send2
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                endtime = MPI_Wtime();
                comm_time += endtime - starttime;

            }
        }

        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        // printf("REACH++?  rank: %d\n", rank);
        MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Trial %d: Elapsed time: %f seconds\n", trial + 1, elapsed_time);
            printf("Trial %d: Communication time: %f seconds\n", trial + 1, comm_time);
        }

        total_elapsed_time += elapsed_time;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double avg_elapsed_time = total_elapsed_time / num_trials *1000;
    double avg_comm_time = total_comm_time / num_trials*1000;

    if (rank == 0) {
        printf("Average Elapsed Time for %d trials: %f ms\n", num_trials, avg_elapsed_time);
        printf("Average Communication Time for %d trials: %f ms\n", num_trials, avg_comm_time);
    }

    MPI_Finalize();
    return 0;
}
