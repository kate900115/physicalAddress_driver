  
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
	void* returnAddr = (void*)(tmp + ((interpAddr<<4) | (0xC<<8)));
	return returnAddr;
}


int main(int argc, char *argv[])
{

	int res = -1;

	int fd = open("/dev/v2p2v", O_RDWR, 0);
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

	cpuaddr_t *phy;
	phy = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	phy->paddr = 0;
	
	res = ioctl(fd, IOCTL_V2P, phy);
	
	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	std::cout<<"phys address = "<<phy->paddr<<std::endl;	


	//configure package length
	uint64_t lengthAddr = read_user_reg(0xC);
		
	cpuaddr_t *length;
	length = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	length->paddr = lengthAddr;
	res = ioctl(fd, IOCTL_P2V, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_P2V\n");
		exit(-1);
	}

	void* va = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (va == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va = 0;
	}

	int* len = (int*)getAddrWithOffset(va, 0xC);
	
	*len = 1024;
	printf("add = %p\n", len);
	printf("add = %d\n", *len);
	
	// configure package address
	// and send the physical address to FPGA
	length->paddr = read_user_reg(0xE);
	res = ioctl(fd, IOCTL_P2V, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_P2V\n");
		exit(-1);
	}

	void* va0 = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va0 == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va0 = 0;
	}

	int* PackAddr = (int*)getAddrWithOffset(va0, 0xE);
	*PackAddr = phy->paddr; 

	printf("PackAddr = %p\n", *PackAddr);

	// send magic number to FPGA
	length->paddr = read_user_reg(0x8);
	res = ioctl(fd, IOCTL_P2V, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_P2V\n");
		exit(-1);
	}

	void* va1 = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va1 == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va1 = 0;
	}

	int* magicNum = (int*)getAddrWithOffset(va1, 0x8);
	*magicNum = 123; 
	std::cout<<"the magicNum written in FPGA is "<<*magicNum<<std::endl;

	// read to check if the value is written
	length->paddr = read_user_reg(0x30);
	res = ioctl(fd, IOCTL_P2V, length);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_P2V\n");
		exit(-1);
	}

	void* va2 = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va2 == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
		va2 = 0;
	}

	int* Reg48 = (int*)va2;
	std::cout<<"the value in reg 48 is "<< *Reg48<<"\n";

	// unmmap all allocated address
	munmap(va, 512);
	munmap(va0, 512);	
	munmap(va1, 512);
	munmap(va2, 512);	


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

