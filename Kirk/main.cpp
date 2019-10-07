//
//  main.cpp
//  Kirk
//
//  Created by Ben Snyder on 10/7/19.
//  Copyright © 2019 Ben Snyder. All rights reserved.
//

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

void makeMatrix(double *loc, int size){
    int i, j;
    for(i=0;i<size;i++){
        for(j=0;j<size;j++){
            *(loc+(i*size+j)) = ((double) rand() / (RAND_MAX));
        }
    }
}

void matMul(double *loc, double *mat1, double *mat2, int size){
    int i, j, k;
    for (i=0; i<size; i++){
        for (j=0; j<size; j++){
            *(loc+(i*size+j)) = 0;
            for (k=0; k<size; k++){
                *(loc+(i*size+j)) += *(mat1+(i*size+k)) + *(mat1+(k*size+j));
            }
        }
    }
}

__global__ void matMulKernel(double *d_C, double *d_A, double *d_B, int size){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((row < size) && (col < size)){
        float Pvalue = 0;
        int k;
        for (k=0; k<size; k++){
            Pvalue += *(d_A+(row*size+k)) * *(d_B+(k*size+col));
        }
        *(d_C+(row*size+col)) = Pvalue;
    }
}

void cudaMatMul(double *loc, double *mat1, double *mat2, int size){
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**), &d_A, size*size*sizeof(double));
    cudaMalloc((void**), &d_B, size*size*sizeof(double));
    cudaMalloc((void**), &d_C, size*size*sizeof(double));
    cudaMemcpy(d_A, mat1, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, mat2, size*size*sizeof(double), cudaMemcpyHostToDevice);
    matMulKernel<<<ceil(size/256.0), 256>>>(d_C, d_A, d_B, size);
    cudaMemcpy(loc, d_C, size*size*sizeof(double), cudaMemcpyDeviceToHost);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int size = 2048;
    double *mat1 = (double*)malloc(size*size*sizeof(double));
    double *mat2 = (double*)malloc(size*size*sizeof(double));
    double *res = (double*)malloc(size*size*sizeof(double));
    makeMatrix(mat1, size);
    makeMatrix(mat2, size);
    double time_spent = 0.0;
    clock_t begin = clock();
    matMul(res, mat1, mat2, size);
    //cudaMatMul(res, mat1, mat2, size);
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    int i, j;
    double acc = 0;
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            acc += *(res+(i*size+j));
        }
    }
    printf("%0.5f\n", acc);
    printf("Time elpased is %f seconds\n", time_spent);
    return 0;
}
