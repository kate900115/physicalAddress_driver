#ifndef __FPGAHealthLib__
#define __FPGAHealthLib__

#include "FPGACoreLib.h"

//
// The FPGA Health API exposes health monitoring information
// specific to the shell.
//

#define  _CDLLEXPORT_

//
// FPGA Health Property IDs.
//
#define FPGA_HEALTH_PROPERTY_CONFIGURATION                0
#define FPGA_HEALTH_PROPERTY_CLOCKRESET                   1
#define FPGA_HEALTH_PROPERTY_TEMPERATURE                  2
#define FPGA_HEALTH_PROPERTY_NETWORK                      3
#define FPGA_HEALTH_PROPERTY_PCIE                         4
#define FPGA_HEALTH_PROPERTY_DRAM                         5
#define FPGA_HEALTH_NUM_PROPERTIES                        6

// 
// Health Severities.
//
#define FPGA_HEALTH_SEVERITY_OK                           0 
#define FPGA_HEALTH_SEVERITY_WARNING                      1
#define FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT              2 
#define FPGA_HEALTH_SEVERITY_ERROR_UNKNOWN                3 
#define FPGA_HEALTH_SEVERITY_ERROR_RMA                    4

#define FPGA_HEALTH_NAME_LEN                              32
#define FPGA_HEALTH_MESSAGE_LEN                           2048

typedef struct
{
    DWORD id;
    char name[FPGA_HEALTH_NAME_LEN];
    char message[FPGA_HEALTH_MESSAGE_LEN];
    DWORD severity;
} FPGA_HEALTH_PROPERTY;

_CDLLEXPORT_ FPGA_STATUS FPGA_GetHealthProperty(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD healthPropertyId, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty);

#endif
