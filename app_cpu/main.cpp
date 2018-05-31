  
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

uint64_t interpAddr = 0x8;
uint64_t FPGA_BASE = 0x90000000;
uint64_t REG_OFFSET = 0x800000;
	

uint64_t read_user_reg( uint64_t usr_reg_addr){
	uint64_t physAddr = FPGA_BASE + ((interpAddr << 4) | (usr_reg_addr));
	return physAddr;
}



int main(int argc, char *argv[])
{
	cpuaddr_t *vaddr;

	int res = -1;
	//unsigned count=0x0A000000;

	int fd = open("/dev/v2p2v", O_RDWR, 0);
	//int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
		return -1;
	}	


	/*----------test-----------*/

	void* address = mmap(0, 512, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

	if (address==NULL){
		fprintf(stderr, "address==NULL\n");
		exit(-1);
	}



	cpuaddr_t *test;
	
	test = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
//	int *addr1;
//	addr1 = (int*) malloc(sizeof(int));
//	test->handle = (void*)addr1;
//
	test->paddr = 0;

	res = ioctl(fd, IOCTL_V2P, test); 
	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	std::cout<<"virtual to physical"<<std::endl;
	std::cout<<"physical address = "<<test->paddr<<std::endl;

	/*----------test----------*/

	

	cpuaddr_t *test_p2v;
	test_p2v = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	//test_p2v->handle = NULL;
	test_p2v->paddr = test->paddr;
	res = ioctl(fd, IOCTL_P2V, test_p2v);
	//res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, test_p2v);
	//res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, test_p2v);
	//res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, test_p2v);


	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	void* va = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va == MAP_FAILED){
		fprintf(stderr,"%s():%s\n", __FUNCTION__, strerror(errno));
	}

	
	int* num = (int*) va;
	*num = 123;

	//*virt_addr = 1;

	std::cout<<"value = "<<*num<<std::endl;


	/*----------test-----------*/

/*
	//configure package length
	uint64_t lengthAddr = read_user_reg(0xC);
	cpuaddr_state_t *length;
	length = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
	length->handle = NULL;
	length->paddr = lengthAddr;
	res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	uint64_t* vAddr = (uint64_t*)length->handle;
	*vAddr = 0x100;
*/
	//while(1);	
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

