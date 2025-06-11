/****************************************************************************


   gcc -O1 -std=gnu11 test_sor_mt.c -lpthread -lrt -lm -o test_SOR_mt

*/

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
#include "apple_pthread_barrier.h"
#endif /* __APPLE__ */

#define CPNS                                                                   \
    3.0 /* Cycles per nanosecond -- Adjust to your computer,                   \
           for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2 /* 2 extra rows/columns for "ghost zone". */

#define A 0    /* coefficient of x^2 */
#define B 10   /* coefficient of x */
#define C 1300 /* constant term */

#define NUM_TESTS 5

/* A, B, and C needs to be a multiple of your BLOCK_SIZE,
   total array size will be (GHOST + Ax^2 + Bx + C) */

#define BLOCK_SIZE 1 // TO BE DETERMINED

#define NUM_THREADS 8
#define OPTIONS 4

#define MINVAL 0.0
#define MAXVAL 10.0

#define TOL 0.00001
#define OMEGA 1.95 // TO BE DETERMINED

typedef double data_t;

typedef struct
{
    long int rowlen;
    data_t* data;
} arr_rec, *arr_ptr;

typedef struct
{
    arr_ptr v;
    int* iterations;
    int thread_id;
    int num_threads;
    pthread_barrier_t* barrier;
    double* local_changes;
} SOR_thread_args;

/* Prototypes */
arr_ptr new_array(long int row_len);
int set_arr_rowlen(arr_ptr v, long int index);
long int get_arr_rowlen(arr_ptr v);
int init_array(arr_ptr v, long int row_len);
int init_array_rand(arr_ptr v, long int row_len);
int print_array(arr_ptr v);

void SOR(arr_ptr v, int* iterations);
void SOR_redblack(arr_ptr v, int* iterations);
void SOR_ji(arr_ptr v, int* iterations);
void SOR_blocked(arr_ptr v, int* iterations);

// multithreaded sor
void* SOR_st(void* threadarg);
void SOR_mt(arr_ptr v, int* iterations);
void* SOR_ji_st(void* threadarg);
void SOR_ji_mt(arr_ptr v, int* iterations);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:

        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_REALTIME, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_REALTIME, &time_stop);
      measurement = interval(time_start, time_stop);

 */

/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/*****************************************************************************/
int main(int argc, char* argv[])
{
    int OPTION;
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    int convergence[OPTIONS][NUM_TESTS];
    int* iterations;

    long int x, n;
    long int alloc_size;

    x = NUM_TESTS - 1;
    alloc_size = GHOST + A * x * x + B * x + C;

    printf("SOR serial variations \n");

    printf("OMEGA = %0.2f\n", OMEGA);

    /* declare and initialize the array */
    arr_ptr v0 = new_array(alloc_size);

    /* Allocate space for return value */
    iterations = (int*)malloc(sizeof(int));

    /* declare and initialize the array */

    OPTION = 0;
    printf("OPTION=%d serial SOR\n", OPTION);
    for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
         x++)
    {
        printf("  iter %d rowlen = %d\n", x, GHOST + n);
        init_array_rand(v0, GHOST + n);
        set_arr_rowlen(v0, GHOST + n);
        clock_gettime(CLOCK_REALTIME, &time_start);
        SOR(v0, iterations);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
        convergence[OPTION][x] = *iterations;
    }

    OPTION++;
    printf("OPTION=%d (mt SOR)\n", OPTION);
    for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
         x++)
    {
        printf("  iter %d rowlen = %d\n", x, GHOST + n);
        init_array_rand(v0, GHOST + n);
        set_arr_rowlen(v0, GHOST + n);
        clock_gettime(CLOCK_REALTIME, &time_start);
        SOR_mt(v0, iterations);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
        convergence[OPTION][x] = *iterations;
    }

    OPTION++;
    printf("OPTION=%d serial SOR ji\n", OPTION);
    for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
         x++)
    {
        printf("  iter %d rowlen = %d\n", x, GHOST + n);
        init_array_rand(v0, GHOST + n);
        set_arr_rowlen(v0, GHOST + n);
        clock_gettime(CLOCK_REALTIME, &time_start);
        SOR_ji(v0, iterations);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
        convergence[OPTION][x] = *iterations;
    }

    OPTION++;
    printf("OPTION=%d (mt SOR ji)\n", OPTION);
    for (x = 0; x < NUM_TESTS && (n = A * x * x + B * x + C, n <= alloc_size);
         x++)
    {
        printf("  iter %d rowlen = %d\n", x, GHOST + n);
        init_array_rand(v0, GHOST + n);
        set_arr_rowlen(v0, GHOST + n);
        clock_gettime(CLOCK_REALTIME, &time_start);
        SOR_ji_mt(v0, iterations);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][x] = interval(time_start, time_stop);
        convergence[OPTION][x] = *iterations;
    }

    printf("All times are in cycles (if CPNS is set correctly in code)\n");
    printf("\n");
    printf("size, serial SOR, SOR iters, mt SOR time, mt SOR iters, "
           "serial SOR ji, SOR ji iters, mt SOR ji time, mt SOR ji iters\n");
    {
        int i, j;
        for (i = 0; i < NUM_TESTS; i++)
        {
            printf("%4d", A * i * i + B * i + C);
            for (OPTION = 0; OPTION < OPTIONS; OPTION++)
            {
                printf(", %10.4g",
                       (double)CPNS * 1.0e9 * time_stamp[OPTION][i]);
                printf(", %4d", convergence[OPTION][i]);
            }
            printf("\n");
        }
    }

} /* end main */

/*********************************/

/* Create 2D array of specified length per dimension */
arr_ptr new_array(long int row_len)
{
    long int i;

    /* Allocate and declare header structure */
    arr_ptr result = (arr_ptr)malloc(sizeof(arr_rec));
    if (!result)
    {
        return NULL; /* Couldn't allocate storage */
    }
    result->rowlen = row_len;

    /* Allocate and declare array */
    if (row_len > 0)
    {
        data_t* data = (data_t*)calloc(row_len * row_len, sizeof(data_t));
        if (!data)
        {
            free((void*)result);
            printf("\n COULDN'T ALLOCATE STORAGE \n", result->rowlen);
            return NULL; /* Couldn't allocate storage */
        }
        result->data = data;
    }
    else
        result->data = NULL;

    return result;
}

/* Set row length of array */
int set_arr_rowlen(arr_ptr v, long int row_len)
{
    v->rowlen = row_len;
    return 1;
}

/* Return row length of array */
long int get_arr_rowlen(arr_ptr v) { return v->rowlen; }

/* initialize 2D array with incrementing values (0.0, 1.0, 2.0, 3.0, ...) */
int init_array(arr_ptr v, long int row_len)
{
    long int i;

    if (row_len > 0)
    {
        v->rowlen = row_len;
        for (i = 0; i < row_len * row_len; i++)
        {
            v->data[i] = (data_t)(i);
        }
        return 1;
    }
    else
        return 0;
}

/* initialize array with random data */
int init_array_rand(arr_ptr v, long int row_len)
{
    long int i;
    double fRand(double fMin, double fMax);

    /* Since we're comparing different algorithms (e.g. blocked, threaded
       with stripes, red/black, ...), it is more useful to have the same
       randomness for any given array size */
    srandom(row_len);
    if (row_len > 0)
    {
        v->rowlen = row_len;
        for (i = 0; i < row_len * row_len; i++)
        {
            v->data[i] = (data_t)(fRand((double)(MINVAL), (double)(MAXVAL)));
        }
        return 1;
    }
    else
        return 0;
}

/* print all elements of an array */
int print_array(arr_ptr v)
{
    long int i, j, row_len;

    row_len = v->rowlen;
    printf("row length = %ld\n", row_len);
    for (i = 0; i < row_len; i++)
    {
        for (j = 0; j < row_len; j++)
        {
            printf("%.4f ", (data_t)(v->data[i * row_len + j]));
        }
        printf("\n");
    }
}

data_t* get_array_start(arr_ptr v) { return v->data; }

double fRand(double fMin, double fMax)
{
    double f = (double)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/************************************/
/* SOR serial */
void SOR(arr_ptr v, int* iterations)
{
    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t* data = get_array_start(v);
    double change, total_change = 1.0e10; /* start w/ something big */
    int iters = 0;

    while ((total_change / (double)(rowlen * rowlen)) > (double)TOL)
    {
        iters++;
        total_change = 0;
        for (i = 1; i < rowlen - 1; i++)
        {
            for (j = 1; j < rowlen - 1; j++)
            {
                change =
                    data[i * rowlen + j] -
                    .25 * (data[(i - 1) * rowlen + j] +
                           data[(i + 1) * rowlen + j] +
                           data[i * rowlen + j + 1] + data[i * rowlen + j - 1]);
                data[i * rowlen + j] -= change * OMEGA;
                if (change < 0)
                {
                    change = -change;
                }
                total_change += change;
            }
        }
        if (abs(data[(rowlen - 2) * (rowlen - 2)]) > 10.0 * (MAXVAL - MINVAL))
        {
            printf("SOR: SUSPECT DIVERGENCE iter = %ld\n", iters);
            break;
        }
    }
    *iterations = iters;
    printf("    SOR() done after %d iters\n", iters);
}

/* SOR for single thread*/
void* SOR_st(void* threadarg)
{
    SOR_thread_args* args = (SOR_thread_args*)threadarg;
    arr_ptr v = args->v;
    int* iterations = args->iterations;
    int tid = args->thread_id;

    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t* data = get_array_start(v);
    // double change, total_change = 1.0e10; /* start w/ something big */
    // int iters = 0;

    // parition work between threads
    int total_update_rows = rowlen - 2; // 2 for ghost rows
    int chunk = total_update_rows / NUM_THREADS;
    int start = 1 + chunk * tid;
    int end = (tid == NUM_THREADS - 1)
                  ? rowlen - 1
                  : start + chunk; // if last row (ghost row), exclude

    double local_change, global_change;
    int local_iter = 0;

    while (1)
    {
        local_change = 0.0;
        for (i = start; i < end; ++i)
        {
            for (j = 1; j < rowlen - 1; ++j)
            {
                double old_val = data[i * rowlen + j];
                double new_val =
                    old_val -
                    OMEGA * (old_val - .25 * (data[(i - 1) * rowlen + j] +
                                              data[(i + 1) * rowlen + j] +
                                              data[i * rowlen + j + 1] +
                                              data[i * rowlen + j - 1]));
                data[i * rowlen + j] = new_val;
                local_change += fabs(new_val - old_val);
            }
        }
        args->local_changes[tid] = local_change;
        pthread_barrier_wait(args->barrier);

        if (tid == 0)
        {
            global_change = 0.0;
            for (i = 0; i < NUM_THREADS; i++)
            {
                global_change += args->local_changes[i];
            }
            *(args->iterations) = local_iter;
            args->local_changes[0] = global_change; // store calculated the
                                                    // total change in slot 0
        }
        pthread_barrier_wait(args->barrier);
        global_change = args->local_changes[0]; // so every thread has
                                                // access to this value

        if ((global_change / (double)(rowlen * rowlen)) <= TOL)
        {
            break;
        }

        local_iter++;
        pthread_barrier_wait(args->barrier);
    }
    printf("thread %d: SOR() done after %d iters\n", tid, local_iter);
    return NULL;
}

void SOR_mt(arr_ptr v, int* iterations)
{
    pthread_t threads[NUM_THREADS];
    SOR_thread_args args[NUM_THREADS];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    // holds changes in each thread
    double* local_changes = malloc(sizeof(double) * NUM_THREADS);
    int rc;
    for (long t = 0; t < NUM_THREADS; ++t)
    {
        args[t].v = v;
        args[t].iterations = iterations;
        args[t].thread_id = t;
        args[t].num_threads = NUM_THREADS;
        args[t].barrier = &barrier;
        args[t].local_changes = local_changes;

        rc = pthread_create(&threads[t], NULL, SOR_st, (void*)&args[t]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (long t = 0; t < NUM_THREADS; ++t)
    {
        if (pthread_join(threads[t], NULL))
        {
            printf("ERROR; code on return from join is %d\n", rc);
            exit(-1);
        }
    }

    pthread_barrier_destroy(&barrier);
    free(local_changes);
}

/* SOR with reversed indices */
void SOR_ji(arr_ptr v, int* iterations)
{
    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t* data = get_array_start(v);
    double change, total_change = 1.0e10; /* start w/ something big */
    int iters = 0;

    while ((total_change / (double)(rowlen * rowlen)) > (double)TOL)
    {
        iters++;
        total_change = 0;
        for (j = 1; j < rowlen - 1; j++)
        {
            for (i = 1; i < rowlen - 1; i++)
            {
                change =
                    data[i * rowlen + j] -
                    .25 * (data[(i - 1) * rowlen + j] +
                           data[(i + 1) * rowlen + j] +
                           data[i * rowlen + j + 1] + data[i * rowlen + j - 1]);
                data[i * rowlen + j] -= change * OMEGA;
                if (change < 0)
                {
                    change = -change;
                }
                total_change += change;
            }
        }
        if (abs(data[(rowlen - 2) * (rowlen - 2)]) > 10.0 * (MAXVAL - MINVAL))
        {
            printf("SOR_ji: SUSPECT DIVERGENCE iter = %d\n", iters);
            break;
        }
    }
    *iterations = iters;
    printf("    SOR_ji() done after %d iters\n", iters);
}

/* SOR for single thread*/
void* SOR_ji_st(void* threadarg)
{
    SOR_thread_args* args = (SOR_thread_args*)threadarg;
    arr_ptr v = args->v;
    int* iterations = args->iterations;
    int tid = args->thread_id;

    long int i, j;
    long int rowlen = get_arr_rowlen(v);
    data_t* data = get_array_start(v);
    // double change, total_change = 1.0e10; /* start w/ something big */
    // int iters = 0;

    // parition work between threads
    int total_update_rows = rowlen - 2; // 2 for ghost rows
    int chunk = total_update_rows / NUM_THREADS;
    int start = 1 + chunk * tid;
    int end = (tid == NUM_THREADS - 1)
                  ? rowlen - 1
                  : start + chunk; // if last row (ghost row), exclude

    double local_change, global_change;
    int local_iter = 0;

    while (1)
    {
        local_change = 0.0;
        for (j = start; j < end; ++j)
        {
            for (i = 1; i < rowlen - 1; ++i)
            {
                double old_val = data[i * rowlen + j];
                double new_val =
                    old_val -
                    OMEGA * (old_val - .25 * (data[(i - 1) * rowlen + j] +
                                              data[(i + 1) * rowlen + j] +
                                              data[i * rowlen + j + 1] +
                                              data[i * rowlen + j - 1]));
                data[i * rowlen + j] = new_val;
                local_change += fabs(new_val - old_val);
            }
        }
        args->local_changes[tid] = local_change;
        pthread_barrier_wait(args->barrier);

        if (tid == 0)
        {
            global_change = 0.0;
            for (i = 0; i < NUM_THREADS; i++)
            {
                global_change += args->local_changes[i];
            }
            *(args->iterations) = local_iter;
            args->local_changes[0] = global_change; // store calculated the
                                                    // total change in slot 0
        }
        pthread_barrier_wait(args->barrier);
        global_change = args->local_changes[0]; // so every thread has
                                                // access to this value

        if ((global_change / (double)(rowlen * rowlen)) <= TOL)
        {
            break;
        }

        local_iter++;
        pthread_barrier_wait(args->barrier);
    }
    printf("thread %d: SOR_ji() done after %d iters\n", tid, local_iter);
    return NULL;
}

void SOR_ji_mt(arr_ptr v, int* iterations)
{
    pthread_t threads[NUM_THREADS];
    SOR_thread_args args[NUM_THREADS];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    // holds changes in each thread
    double* local_changes = malloc(sizeof(double) * NUM_THREADS);
    int rc;
    for (long t = 0; t < NUM_THREADS; ++t)
    {
        args[t].v = v;
        args[t].iterations = iterations;
        args[t].thread_id = t;
        args[t].num_threads = NUM_THREADS;
        args[t].barrier = &barrier;
        args[t].local_changes = local_changes;

        rc = pthread_create(&threads[t], NULL, SOR_ji_st, (void*)&args[t]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (long t = 0; t < NUM_THREADS; ++t)
    {
        if (pthread_join(threads[t], NULL))
        {
            printf("ERROR; code on return from join is %d\n", rc);
            exit(-1);
        }
    }

    pthread_barrier_destroy(&barrier);
    free(local_changes);
}