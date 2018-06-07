  
#include "cuda.h"
//#include "cuda_runtime_api.h"
#include "v2p2vioctl.h"

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
#include <ctime>
#include <chrono>

//-----------------------------------------------------------------------------

void checkError(CUresult status);

//-----------------------------------------------------------------------------

uint64_t interpAddr = 0x8;
uint64_t FPGA_BASE = 0x90000000;
	

uint64_t read_user_reg( uint64_t usr_reg_addr){
	uint64_t physAddr = FPGA_BASE + ((interpAddr << 4) | (usr_reg_addr<<8));
	return physAddr;
}


void* getAddrWithOffset(void* addr, uint64_t usr_reg_addr){
	char* tmp = (char*) addr;
	void* returnAddr = (void*)(tmp + ((interpAddr<<4) | (usr_reg_addr<<8)));
	return returnAddr;
}


int main(int argc, char *argv[])
{

	int res = -1;

	int fd = open("/dev/v2p2v", O_RDWR, 0);
	//int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/v2p2v");
		return -1;
	}

	// allocate a variable on CPU and then get its physical address
	void* cpu_va = mmap(0,512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (cpu_va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	printf("virtual addr: %p\n", cpu_va);

	cpuaddr_t *phy;
	phy = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	phy->paddr = 0;
	
	res = ioctl(fd, IOCTL_V2P, phy);
	
	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	printf("phys address = %p\n", phy->paddr);	
	printf("FPGA haven't written the CPU memory: %lx\n", *(int*)cpu_va);


	//configure package length
	uint64_t lengthAddr = read_user_reg(0x8);
		
	cpuaddr_t *length;
	length = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	length->paddr = lengthAddr;
	res = ioctl(fd, IOCTL_P2V, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_P2V\n");
		exit(-1);
	}

	void* va = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	printf("virtual address = %p\n", va);
	printf("physical address = %p\n", length->paddr);

	if (va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va = 0;
	}

	volatile uint64_t* reset = (uint64_t*)getAddrWithOffset(va, 0x0);
	*reset = 1;

	volatile uint64_t *read_data = (uint64_t*)getAddrWithOffset(va, 0xC);
	printf("reset is done.\n");
	printf("write times = %lu\n", *read_data);



	// unmmap all allocated address
	munmap(va, 512);


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

