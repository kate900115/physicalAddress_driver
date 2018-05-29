  
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

	int res = -1;
	//unsigned count=0x0A000000;

	int fd = open("/dev/Catapult FPGA", O_RDWR, 0);
	//int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
		return -1;
	}

	// allocate a variable on CPU and then get its physical address
	void* cpu_va = mmap(0,512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (cpu_va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	cpuaddr_state_t *phy;
	phy = (struct cpuaddr_state_t*)malloc(sizeof(struct cpuaddr_state_t));
	phy->handle = NULL;
	phy->paddr = 0;
	
	res = ioctl(fd, IOCTL_GPUMEM_LOCK, phy);
	
	if (res<0){
		fprintf(stderr, "Error in IOCTL_LOCK\n");
		exit(-1);
	}

	std::cout<<"phys address = "<<phy->paddr<<std::endl;	


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

	void* va = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	
	if (va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va = 0;
	}

	int* add = (int*) va;
	*add = 1024;

	
	// configure package address
	lengthAddr = read_user_reg(0xE);
	length->handle = NULL;
	length->paddr = lengthAddr;
	res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	void* va0 = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va0 == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va0 = 0;
	}

	int* add0 = (int*)va0;
	*add0 = phy->paddr; 

	// magic number
	lengthAddr = read_user_reg(0x8);
	length->handle = NULL;
	length->paddr = lengthAddr;
	res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
		exit(-1);
	}

	void* va1 = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va1 == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va1 = 0;
	}

	int* magicNum = (int*)va1;
	*magicNum = 123; 

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

