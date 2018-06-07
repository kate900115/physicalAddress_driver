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

#include "cutemp/vadd.h"

// time measurement
#include <chrono>
#include <ctime>
#include <iostream>
//-----------------------------------------------------------------------------

void checkError(CUresult status);
bool wasError(CUresult status);

//-----------------------------------------------------------------------------

// zyuxuan: struct for pthread parameter
// zyuxuan: function for the thread

int main(int argc, char *argv[])
{
    gpudma_lock_t lock;
    gpudma_unlock_t unlock;
    gpudma_state_t *state = 0;
    int statesize = 0;
    int res = -1;
    unsigned count=0x0A000000;

    int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
    if (fd < 0) {
        printf("Error open file %s\n", "/dev/"GPUMEM_DRIVER_NAME);
        return -1;
    }

    int fd_v2p2v = open("/dev/v2p2v", O_RDWR, 0);
    if (fd_v2p2v < 0){
	printf("Error open file %s\n","/dev/v2p2v");
	return -1;
    }

    checkError(cuInit(0));

    int total = 0;
    checkError(cuDeviceGetCount(&total));
    fprintf(stderr, "Total devices: %d\n", total);

    CUdevice device;
    checkError(cuDeviceGet(&device, 0));

    char name[256];
    checkError(cuDeviceGetName(name, 256, device));
    fprintf(stderr, "Select device: %s\n", name);

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    checkError( cuDeviceComputeCapability(&major, &minor, device));
    fprintf(stderr, "Compute capability: %d.%d\n", major, minor);

    size_t global_mem = 0;
    checkError( cuDeviceTotalMem(&global_mem, device));
    fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(global_mem >> 20));
    if(global_mem > (unsigned long long)4*1024*1024*1024L)
        fprintf(stderr, "64-bit Memory Address support\n");

    CUcontext  context;
    checkError(cuCtxCreate(&context, CU_CTX_MAP_HOST, device));


	// zyuxuan: allocate memory space on host and device
	CUdeviceptr d_a, d_b, d_c;
	float h_a[m*n], h_b[m*n], h_c[m*n];

	for (int i=0; i<m*n; i++){
		h_a[i] = i;
		h_b[i] = i*i;
		h_c[i] = 0;
	}

	// zyxuan: setup device memory
	checkError(cuMemAlloc(&d_a,sizeof(int)*m*n));
	checkError(cuMemAlloc(&d_b,sizeof(int)*m*n));
	checkError(cuMemAlloc(&d_c,sizeof(int)*m*n));

	// zyuxuan: offload data from CPU to GPU
	checkError(cuMemcpyHtoD(d_a, h_a, sizeof(int)*m*n));
	checkError(cuMemcpyHtoD(d_b, h_b, sizeof(int)*m*n));




	// zyuxuan: to insert our cuda function
	// zyuxuan: we need to load module 
	char* module_file = (char*) "cutemp/vadd.ptx";
	char* kernel_name = (char*) "vadd";
	CUmodule module;
	
	checkError(cuModuleLoad(&module, module_file));

	CUfunction function;
	checkError(cuModuleGetFunction(&function, module, kernel_name));

	// zyuxuan: to allow GPU to access to CPU memory space
	void* p;
	p = (int*)malloc(20*sizeof(int));
	checkError(cuMemHostRegister(p, 20*sizeof(int), CU_MEMHOSTREGISTER_DEVICEMAP));
	CUdeviceptr CPUflag;
	checkError(cuMemHostGetDevicePointer(&CPUflag, p,0));

	int* p_flag = (int*) p;


    size_t size = 0x100000;
    CUdeviceptr dptr = 0;
    unsigned int flag = 1;
    unsigned char *h_odata = NULL;
    h_odata = (unsigned char *)malloc(size);

    CUresult status = cuMemAlloc(&dptr, size);

	/* this code does not work */
	//CUresult status = cuModuleGetGlobal(&dptr, &size, module, "d_lock");
	//size = 0x100000;

    if(wasError(status)) {
        goto do_free_context;
    }

    fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr);

    status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    if(wasError(status)) {
        goto do_free_memory;
    }

    fprintf(stderr, "Press enter to lock\n");
    //getchar();

	// TODO: add kernel driver interaction...
    lock.addr = dptr;
    lock.size = size;
    res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock);
    if(res < 0) {
        fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_LOCK\n");
        goto do_free_attr;
    }

    fprintf(stderr, "Press enter to get state. We lock %ld pages\n", lock.page_count);
    //getchar();

    statesize = (lock.page_count*sizeof(uint64_t) + sizeof(struct gpudma_state_t));
    state = (struct gpudma_state_t*)malloc(statesize);
    if(!state) {
        goto do_free_attr;
    }
    memset(state, 0, statesize);
    state->handle = lock.handle;
    state->page_count = lock.page_count;
    res = ioctl(fd, IOCTL_GPUMEM_STATE, state);
    if(res < 0) {
        fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
        goto do_unlock;
    }

    fprintf(stderr, "Page count 0x%lx\n", state->page_count);
    fprintf(stderr, "Page size 0x%lx\n", state->page_size);

    for(unsigned i=0; i<state->page_count; i++) {
        fprintf(stderr, "%02d: 0x%lx\n", i, state->pages[i]);
        //void* va = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[i]);
        void* va = mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[0]);
        if (va == MAP_FAILED ) {
             fprintf(stderr, "%s(): %s\n", __FUNCTION__, strerror(errno));
             va = 0;
        } 
	else {
            	//memset(va, 0x55, state->page_size);
        	unsigned *ptr=(unsigned*)va;
        	for( unsigned jj=0; jj<(state->page_size/4); jj++ )
        	{
        		*ptr++=count++;
        	}

            	fprintf(stderr, "%s(): Physical Address 0x%lx -> Virtual Address %p\n", __FUNCTION__, state->pages[i], va);


			// zyuxuan: let's assume at this point we get both virtual pointer (CUdeviceptr dptr)
			// zyuxuan: and the physical pointer (void* va) 
		int* tmpPointer = (int*) va;
		*tmpPointer = 2;
		
		//sleep (1);
		*p_flag = 1;
		// launch kernel
		void* args[5] = {&d_a, &d_b, &d_c, &dptr, &CPUflag};
		checkError(cuLaunchKernel(function, m, n, 1, 16, 16, 1, 0, 0, args,0));
		printf("kernel launched!\n");
			
		int countNum = 0;
		auto start = std::chrono::high_resolution_clock::now();
		while(countNum<10000){
			
		//printf("CPU: set lock to be zero!\n%d", countNum);
			
		*tmpPointer = 0;

		//cuMemcpyDtoH(h_c, d_c, sizeof(int)*m*n);
		//for (int i=0; i<m*n; i++){
				//	printf("h_c[%d]= %f,",i,h_c[i]);
				//	if (i%5==4) printf("\n");
				//}

				while (*p_flag);
				*p_flag = 1;
				//printf("CPU: flag is set to be 1.\n");
				//
				//
				//cuMemcpyDtoH(h_c, d_c, sizeof(int)*m*n);
				//for (int i=0; i<m*n; i++){
				//	printf("h_c[%d]= %f,",i,h_c[i]);
				//	if (i%5==4) printf("\n");
				//}


				countNum++;
			}
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> diff = end - start;
			std::cout<<"it took me "<<diff.count()<<" seconds."<<std::endl;
			cuCtxSynchronize();
		    munmap(va, state->page_size);
        }
    }

    {
        //const void* d_idata = (const void*)dptr;
    	//cudaMemcpy(h_odata, d_idata, size, cudaMemcpyDeviceToHost);
    	//cudaDeviceSynchronize();

    	cuMemcpyDtoH( h_odata, dptr, size );
    	cuCtxSynchronize();

    	unsigned *ptr = (unsigned*)h_odata;
    	unsigned val;
    	unsigned expect_data=0x0A000000;
    	unsigned cnt=size/4;
    	unsigned error_cnt=0;
    	for( unsigned ii=0; ii<cnt; ii++ )
    	{
    		val=*ptr++;
    		if( val!=expect_data )
    		{
    			error_cnt++;
    			if( error_cnt<32 )
    			 fprintf(stderr, "%4d 0x%.8X - Error  expect: 0x%.8X\n", ii, val, expect_data );
    		} else if( ii<16 )
    		{
      		  fprintf(stderr, "%4d 0x%.8X \n", ii, val );
    		}
    		expect_data++;

    	}
    	if( 0==error_cnt )
    	{
    		  fprintf(stderr, "\nTest successful\n" );
    	} else
    	{
    		  fprintf(stderr, "\nTest with error\n" );
    	}
    }


    fprintf(stderr, "Press enter to unlock\n");
    //getchar();

do_unlock:
    unlock.handle = lock.handle;
    res = ioctl(fd, IOCTL_GPUMEM_UNLOCK, &unlock);
    if(res < 0) {
        fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");
        goto do_free_state;
    }
do_free_state:
    free(state);
do_free_attr:
    flag = 0;
    cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);

do_free_memory:
    cuMemFree(dptr);

do_free_context:
    cuCtxDestroy(context);

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
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        exit(0);
    }
}

//-----------------------------------------------------------------------------

bool wasError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------
