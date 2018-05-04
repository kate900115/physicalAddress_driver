  
#include "cuda.h"
//#include "cuda_runtime_api.h"
#include "gpumemioctl.h"

#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>

// zyuxuan
#include <iostream>

// zyuxuan for FPGA
#include "FPGA/FPGACoreLib.h"
#include "FPGA/FPGACoreLib.cpp"
#include "FPGA/FPGAHealthLib.h"
#include "FPGA/FPGAHealthLib.cpp"

//-----------------------------------------------------------------------------

void checkError(CUresult status);

//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	//zyuxuan
	cpuaddr_state_t *vaddr;

	int res = -1;
	//unsigned count=0x0A000000;

	int fd = open("/dev/Catapult FPGA", O_RDWR, 0);
	//int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
		return -1;
	}	

	// TODO: add kernel driver interaction...
	// zyuxuan


	vaddr = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
	int *addr;
	addr = (int*) malloc(sizeof(int));
	vaddr->handle = (void*)addr;
	vaddr->paddr = 0;

	res = ioctl(fd, IOCTL_GPUMEM_LOCK, vaddr); 
	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	std::cout<<"physical address = "<<vaddr->paddr<<std::endl;

	close(fd);

	return 0;
}

// -------------------------------------------------------------------

void checkError(CUresult status)
{
	if(status != CUDA_SUCCESS) {
		const char *perrstr = 0;
		CUresult ok = cuGetErrorString(status,&perrstr);
		if(ok == CUDA_SUCCESS) {
			if(perrstr) {
				fprintf(stderr, "info: %s\n", perrstr);
			} 
			else {
				fprintf(stderr, "info: unknown error\n");
			}
		}
		exit(0);
	}
}

//-----------------------------------------------------------------------------

