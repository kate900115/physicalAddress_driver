#include "vadd.h"
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	while(count<1000000000){
		count++;
		if ((i==0)&&(j==0)){
			*flag=*flag+count;
		}
	}
}

