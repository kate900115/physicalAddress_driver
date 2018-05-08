  
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
//#include "FPGA/FPGACoreLib.h"
//#include "FPGA/FPGACoreLib.cpp"
//#include "FPGA/FPGAHealthLib.h"
//#include "FPGA/FPGAHealthLib.cpp"

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


//	vaddr = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
//	int *addr;
//	addr = (int*) malloc(sizeof(int));
//	vaddr->handle = (void*)addr;
//	vaddr->paddr = 0;

	/*----------test-----------*/
	cpuaddr_state_t *test;
	
	test = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
	int *addr1;
	addr1 = (int*) malloc(sizeof(int));
	test->handle = (void*)addr1;
	test->paddr = 0;

	res = ioctl(fd, IOCTL_GPUMEM_LOCK, test); 
	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	std::cout<<"virtual to physical"<<std::endl;
	std::cout<<"physical address = "<<test->paddr<<std::endl;

	/*----------test----------*/

	cpuaddr_state_t *test_p2v;
	test_p2v = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
	test_p2v->handle = NULL;
	test_p2v->paddr = test->paddr;
	res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, test_p2v);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	int* virt_addr = (int*)test_p2v->handle;
	*virt_addr = 1;

	std::cout<<"value = "<<*addr1<<std::endl;
/*
	void* va = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va = 0;
	}
	else{
		int *Addr = (int*) va;
		vaddr->handle = va;
		vaddr->paddr = 0;
		res = ioctl(fd, IOCTL_GPUMEM_LOCK, vaddr);
		if (res<0){
			fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		}
		std::cout<<"physical address = "<<vaddr->paddr<<std::endl;

		munmap(va, 512);
	}
*/


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

