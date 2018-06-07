#include "vadd.h"

__shared__ long shared_sum;
	
extern "C" __global__ void vadd(float *A, float* B, float* C, int* d_lock, volatile int* flag){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int count = 0;
	long sum = 0;
	while(count<1000){
		count++;
		if ((i==0)&&(j==0)){
			//*flag = count;
			//printf("nnnnnnnnnnnnnnnnnnnnnnn\n");
			//int a = *flag;
			//printf("*flag = %d\n", *flag);
			*flag =  count+1;
			int a = *flag + 1;
			//printf("*flag = %d\n", a);
			//shared_sum = count+1;
	//		shared_sum = count+1;
		}
	}
//	*d_lock = shared_sum;
}

