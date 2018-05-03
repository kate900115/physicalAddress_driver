#ifndef __FPGA_CORE_H__
#define __FPGA_CORE_H__

#include <string>

// WARNING: DO NOT CHANGE THESE WITHOUT UPDATING THE CORRESPONDING VALUES IN HARDWARE
// THESE SHOULD BE TREATED AS HARD-CODED VALUES, NOT PARAMETERS.

#define INTER_ADDR_FULL_STATUS_REG                   0
#define INTER_ADDR_DONE_STATUS_REG                   1
#define INTER_ADDR_PEND_STATUS_REG                   2
#define INTER_ADDR_GENERAL_PURPOSE_REG               3
#define INTER_ADDR_PROBE_IN_FPGA_BUFFER_0            4
#define INTER_ADDR_PROBE_IN_FPGA_BUFFER_1            5
#define INTER_ADDR_PROBE_OUT_FPGA_BUFFER_0           6
#define INTER_ADDR_PROBE_OUT_FPGA_BUFFER_1           7
//#define INTER_ADDR_PROBE_RES_FPGA_BUFFER_0           8 -- repurposed
//#define INTER_ADDR_PROBE_RES_FPGA_BUFFER_1           9 -- repurposed
#define INTER_ADDR_ASMI_RSU                         10
#define INTER_ADDR_AVALON                           11
#define INTER_ADDR_HACK_OVERRIDE_OUT_DATA_SIZE      12
#define INTER_ADDR_ENABLE_DISABLE                   13
#define INTER_ADDR_INTERRUPT                        14
#define INTER_ADDR_DMA_DESCRIPTORS_AND_RESERVED     15

// repurposed interpretation address for native 64 bit soft register interface
#define INTER_ADDR_SOFT_REG                          8
#define INTER_ADDR_SOFT_REG_CAPABILITY               9
#define SOFT_REG_CAPABILITY_SIGNATURE                0x50F750F7
#define SOFT_REG_SLOT_DMA_BASE_ADDR                  0x7E00
#define SOFT_REG_SLOT_DMA_MAGIC_ADDR                 (SOFT_REG_SLOT_DMA_BASE_ADDR + 63)
#define SOFT_REG_MAPPING_SLOT_DMA_MAGIC_VALUE        0x8926fc9c4e6256d9ULL
//in hardware, SOFT_REG_MAPPING_SLOT_DMA_MAGIC_VALUE is defined in SoftRegs_Adapter.sv

//hardcoded constants from Jason Thong based on deep knowledge of the slot DMA hardware design, DO NOT MODIFY
//the FPGA buffer size range is largely determined by DMA CPU to FPGA -- it requires at least 2 PCIe tags, but can only use up to 32 tags as limited by the PCIe spec
//tags are hardcoded in size to 4 KB
#define MIN_FPGA_BUFFER_SIZE                        8192
#define MAX_FPGA_BUFFER_SIZE                        131072

//the number of slots must be at least 2 otherwise it breaks the verilog syntax for some multiplexers in hardware
//conceptually the design will support 1 slot but there is no practical point given the FPGA is double buffered
//the software ISR handshaking (32-bit PCIe reads) requires that slot numbers are representable on 8 bits, hence up to 256 can be used
#define MIN_FPGA_NUM_SLOTS                          2
#define MAX_FPGA_NUM_SLOTS                			256

//Mt Granite and Pikes Peak each use up to 128 Shell registers, require that hardware support at least this many
//upper bound comes from the application address only using 16 bits
#define MIN_FPGA_NUM_SHELL_REG                      128
#define MAX_FPGA_NUM_SHELL_REG                      65536


#define ASMI_BYTES_PER_SECTOR					    65536

#define MAX_BYTES_LOG_ERROR                         2048
#define MUTEX_TIMEOUT_MILLISECONDS                  10000

#define	_SUCCESS(X)	_FPGA_TEST_check_status (X)

#define LOG(fpgaHandle, level, format, ...) {\
    if(fpgaHandle) {\
        FPGA_HANDLE_INT *__handle = (FPGA_HANDLE_INT *)(fpgaHandle); \
        if(__handle->logfunc != NULL) {\
            char buf[512];\
            sprintf_s(buf, sizeof(buf), "[hip-%u]: " format, __handle->epnum, ##__VA_ARGS__);\
            __handle->logfunc(level, buf);\
        }\
    }\
}

#define LOG2(logFunc, level, format, ...) {\
    if(logFunc) {\
            char buf[512];\
            sprintf_s(buf, sizeof(buf), format, ##__VA_ARGS__);\
            logFunc(level, buf);\
        }\
}

#define LOG_INFO(fpgaHandle, format, ...) LOG(fpgaHandle, FPGA_LOG_INFO, format, ##__VA_ARGS__);

#define LOG_ERROR(status, errcode, format, ...) {\
    struct timespec ts; timespec_get(&ts, TIME_UTC); \
    char buf[MAX_BYTES_LOG_ERROR]; \
    status = errcode; \
    __lastError = errcode; \
    sprintf_s(buf, "%llu,%llu,%u,%s,%d,%s\r\n", \
                        ts.tv_sec, \
                        ts.tv_nsec, \
                        getpid(), \
                        __FILE__, \
                        __LINE__, \
                        format); \
    sprintf_s(__errMessage, buf, ##__VA_ARGS__); \
}


//
// An FPGA handle is created for each PCIe endpoint.
//

typedef struct 
{
    // Information
    DWORD                       epnum;                          // PCIe endpoint number
    DWORD                       flags;                          // User flags.
    LPVOID                      versions;                       // Pointer to container used for version checking.
    DWORD                       versionMajorMinor;        // HIWORD: major loaded driver version, LOWORD: minor loaded driver version
    void                        (*logfunc)(DWORD logflag, const char *pchString);
    
    // Device driver handle and BAR.
	HANDLE				        device_handle;                  // Device driver handle 
	DWORD				        register_space_size;            // Size of PCIe BAR 0 associated with device handle.
	ULONG_PTR			        bar0_registers;                 // Pointer to BAR 0 associated with device handle.
        ULONG_PTR			        bar0_registers_kvs;                 // Pointer to BAR 0 associated with device handle.  

    // Mutexes
	HANDLE				        mutex_for_enable_disable;       // Provides atomic access to enable/disable interface.
    HANDLE                      mutex_softreg;                  // Provides atomic access to soft register interface.
    HANDLE                      mutex_flash;                    // Provides atomic access to flash interface.

    // Toggles.
	bool					    is_fpga_app_en;                 // Role DMA support enabled.
	bool					    is_dma_pc_to_fpga_en;           // Host-to-FPGA DMAs enabled.
	bool					    is_dma_fpga_to_pc_en;           // FPGA-to-Host DMAs enabled.

    // Buffer information
	DWORD			            num_pc_buffers;					// Number of DMA buffers supported by FPGA -- the number of slots usable by software, can be limited by CPU buffer size
	DWORD			            num_pc_buffers_raw;				// Number of DMA buffers supported by FPGA -- the number of slots physically supported by the FPGA
	DWORD			            fpga_buffer_size_bytes;			// Bytes per DMA buffer supported by FPGA -- the slot size as used by software
	DWORD			            fpga_buffer_size_raw;			// Bytes per DMA buffer supported by FPGA -- the physical size of each DMA buffer on the FPGA
	DWORD		                num_general_purpose_registers;  // Number of general purpose register supported by FPGA.

    // User-mapped buffers
    DWORD                       **input_buffer,
                                **output_buffer,
                                **result_buffer;

    // Buffer physical address bases
	ULONG64                     input_buffer_phys,
                                output_buffer_phys,
                                result_buffer_phys,
                                control_buffer_phys;

	// Buffer size allocated by driver
	ULONG                       input_buffer_bytes,
		                        output_buffer_bytes;

    // Synchronization and interrupt service routine
	volatile DWORD              *ctrl_buffer;                   // Control bits used for FPGA/Host DMA synchronization (FULL, DONE doorbell registers)
	BOOL					    interrupts_on;                  // Set when interrupts are enabled.
	HANDLE                      *isr_wake_up_events;            // array of events, threads go to sleep if they see an "early done", isr wakes up thread when it gets interrupt
    LONGLONG                    ticks_per_second;               // used for time-outs during polling-based synchronization.
    
	// shell capabilities
	bool                        soft_registers_native_64_support;	// whether hardware has native 64-bit support for soft register interface (possible on Abalone PCIe gen3 and Bedrock)
	bool                        soft_registers_mapping_slot_dma;	// if the above is true, are the legacy PCIe registers being mapped over soft registers (possibly on Bedrock)
	
	// PCIe AER & Base Error Reporting
    bool                        pcie_has_AER;
    UINT32                      pcie_saved_AER;
    bool                        pcie_saved_CER;
    bool                        pcie_saved_FER;
    bool                        pcie_saved_NFER;

    // Reconfiguration state
	int				            saved_rootport_inst;
    char                        m_config_space[4096];

	// Rootport interface
	HANDLE                      rootport_handle;
	DWORD                       rootport_instances_count;

    // All RPs Saved State
    bool                        rp_saved_aer_present[128];
    UINT32                      rp_saved_aer[128];
    bool                        rp_saved_CER[128];
    bool                        rp_saved_FER[128];
    bool                        rp_saved_NFER[128];
    DWORD                       rp_saved_num_rps;

    char *                      device_file_path;
} FPGA_HANDLE_INT;

typedef FPGA_HANDLE_INT* FPGA_HANDLE;
#endif
