#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <errno.h>

#include "FPGACoreLib.h"
#include "FPGAInternals.h"
#define _CDLLEXPORT_

char __errMessage[MAX_BYTES_LOG_ERROR] = "No errors";
FPGA_STATUS __lastError = FPGA_STATUS_SUCCESS;

static LONG64 handlesOpenedPerProcess = 0;

_CDLLEXPORT_ FPGA_STATUS FPGA_GetLastErrorText(_Out_ CHAR *pErrorTextBuf, _In_ DWORD cbBufSize)
{
    strcpy_s(pErrorTextBuf, cbBufSize, __errMessage);
    return FPGA_STATUS_SUCCESS;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_SetLastErrorText(_In_ CHAR *pErrorTextBuf)
{
    strcpy_s(__errMessage, MAX_BYTES_LOG_ERROR, pErrorTextBuf);
    return FPGA_STATUS_SUCCESS;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetLastError()
{
    return __lastError;
}

_CDLLEXPORT_ VOID FPGA_SetLastError(FPGA_STATUS err)
{
    __lastError = err;
}

static void _FPGA_DefaultLogger(DWORD log_level, const char *buf)
{
    if ((log_level & FPGA_LOG_ERROR) || (log_level & FPGA_LOG_FATAL) || (log_level & FPGA_LOG_WARN))
    {
        fprintf_s(stderr, "%s", buf);
    }
}

//
// The following two functions are 'hidden' from normal API users, but is required by the Pikes Peak Factory Tester.
//

_CDLLEXPORT_ DWORD64 FPGA_LowLevelRead64(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	DWORD64 rdData;
	if (fpgaHandleInt->soft_registers_mapping_slot_dma) {
		if (interpAddress == INTER_ADDR_DMA_DESCRIPTORS_AND_RESERVED && appAddress <= 53) {
			rdData = _FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + appAddress);
		}
		else {
			//should report an error for unmapped address, not possible unless we change the API 
			rdData = 0xffffffffffffffffULL;	//for now just imitate what a disconnected PCIe would return
		}
	}
	else {
		rdData = _FPGA_low_level_read_64(fpgaHandle, interpAddress, appAddress);
	}
	return rdData;
}

_CDLLEXPORT_ VOID FPGA_LowLevelWrite64(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress, _In_ DWORD64 writeData)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	if (fpgaHandleInt->soft_registers_mapping_slot_dma) {
		if (interpAddress == INTER_ADDR_DMA_DESCRIPTORS_AND_RESERVED && appAddress <= 53) {
			FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
			_FPGA_low_level_write_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + appAddress, writeData);
		}
		//else should report an error for unmapped address, not possible unless we change the API 
	}
	else {
		_FPGA_low_level_write_64(fpgaHandle, interpAddress, appAddress, writeData);
	}
}

//
// End hidden stuff needed for the factory tester
//

_CDLLEXPORT_ FPGA_STATUS FPGA_IsDevicePresent(_In_ const char *pchVerManifestFile, _In_ void(*logFunc)(DWORD logflag, const char *pchString))
{
    FPGA_STATUS status = FPGA_STATUS_HARDWARE_NOT_PRESENT;
    if (logFunc == NULL)
    {
        logFunc = _FPGA_DefaultLogger;
    }

    if (access("/dev/catapult_fpga_hip1", F_OK)) {
        status = FPGA_STATUS_SUCCESS;
    }

finish:
    return status;
}


FPGA_STATUS _FPGA_GetSoftRegisterCapability(_In_ FPGA_HANDLE fpgaHandle)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	fpgaHandleInt->soft_registers_native_64_support = true;
	for (DWORD i = 0; i < 32; i++) {	//legacy slot dma result buffer is 128 bytes
		if (_FPGA_low_level_read_legacy(fpgaHandle, INTER_ADDR_SOFT_REG_CAPABILITY, i) != SOFT_REG_CAPABILITY_SIGNATURE) {
			fpgaHandleInt->soft_registers_native_64_support = false;
			break;
		}
	}
	fpgaHandleInt->soft_registers_mapping_slot_dma = false;
	if (fpgaHandleInt->soft_registers_native_64_support) {
		if (_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_MAGIC_ADDR) == SOFT_REG_MAPPING_SLOT_DMA_MAGIC_VALUE) {
			fpgaHandleInt->soft_registers_mapping_slot_dma = true;
		}
	}
//	printf("fpgaHandleInt->soft_registers_native_64_support %d\n", fpgaHandleInt->soft_registers_native_64_support);
//	printf("fpgaHandleInt->soft_registers_mapping_slot_dma %d\n", fpgaHandleInt->soft_registers_mapping_slot_dma);
	return status;
}

static int find_device_major(int endpointNumber)
{
    char line[256] = {0};
    char module_name[256] = {0};
    char expected_module_name[256] = {CATAPULT_FPGA_DEVICE_NAME_PREFIX};
    sprintf(&expected_module_name[strlen(expected_module_name)], "%d", endpointNumber + 1);
    int device_major = -1;
    FILE * fp = fopen("/proc/devices", "r");
    while (NULL != fgets(line, sizeof(line), fp)) {
        if (2 == sscanf(line, "%d %s", &device_major, module_name)) {
            if (strcmp(module_name, expected_module_name) == 0) {
                fclose(fp);
                return device_major;
            }
        }
    }
    fclose(fp);
    return -1;
}

static FPGA_STATUS map_buffers(FPGA_HANDLE_INT * pFpgaHandleInt, int endpointNumber)
{
    int status = FPGA_STATUS_SUCCESS;
    int dev_major = find_device_major(endpointNumber);
    if (dev_major < 0) {
        LOG_ERROR(status, FPGA_STATUS_ERROR_OPENING_DEVICE, "Unable to find device HIP%d", endpointNumber + 1);
        return status;
    }
    char random_filename[10] = {0};
    srand(time(NULL));
    for (int i=0; i<sizeof(random_filename); i++)
        random_filename[i] = '0' + rand() % 10;
    char dev_file_path[100] = "/dev/catapult_fpga_";
    memcpy(dev_file_path + strlen(dev_file_path), random_filename, sizeof(random_filename));
    if (0 != mknod(dev_file_path, S_IFCHR | S_IRUSR | S_IWUSR, makedev(dev_major, 0))) {
        LOG_ERROR(status, FPGA_STATUS_ERROR_OPENING_DEVICE, "Unable to mknod %s major %d, errno %d (%s)", dev_file_path, dev_major, errno, strerror(errno));
        return status;
    }

    int fd = open(dev_file_path, O_SYNC);
    if (fd < 0) {
        LOG_ERROR(status, FPGA_STATUS_ERROR_OPENING_DEVICE, "Cannot open FPGA device %s major %d, errno %d (%s)", dev_file_path, dev_major, errno, strerror(errno));
        unlink(dev_file_path);
        return status;
    }

    void * input_buffer = mmap(NULL, CATAPULT_FPGA_DATA_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    void * output_buffer = mmap(NULL, CATAPULT_FPGA_DATA_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 1 << CATAPULT_FPGA_MMAP_SHIFT_BITS);
    void * result_buffer = mmap(NULL, CATAPULT_FPGA_META_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 2 << CATAPULT_FPGA_MMAP_SHIFT_BITS);
    pFpgaHandleInt->ctrl_buffer = (DWORD*) mmap(NULL, CATAPULT_FPGA_META_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 3 << CATAPULT_FPGA_MMAP_SHIFT_BITS);
    pFpgaHandleInt->bar0_registers = (ULONG_PTR) mmap(NULL, CATAPULT_FPGA_BAR0_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 4 << CATAPULT_FPGA_MMAP_SHIFT_BITS);
    printf("bar0_registers %p errno %d\n", pFpgaHandleInt->bar0_registers, errno);

    if (input_buffer == MAP_FAILED
        || output_buffer == MAP_FAILED
        || result_buffer == MAP_FAILED
        || pFpgaHandleInt->ctrl_buffer == MAP_FAILED
        || (PVOID)pFpgaHandleInt->bar0_registers == MAP_FAILED) {

	//zyuxuan
	if (input_buffer==MAP_FAILED) {printf("input buffer failed!\n");}
	if (output_buffer==MAP_FAILED) {printf("output buffer failed!\n");}
	if (result_buffer==MAP_FAILED) {printf("result buffer failed!\n");}
	if (pFpgaHandleInt->ctrl_buffer == MAP_FAILED) {printf("ctrl buffer failed!\n");}
	if ((PVOID)pFpgaHandleInt->bar0_registers==MAP_FAILED) {printf("bar register failed!\n");}



        LOG_ERROR(status, FPGA_STATUS_ERROR_OPENING_DEVICE, "Failed to map buffers in FPGA device %s major %d, errno %d (%s)", dev_file_path, dev_major, errno, strerror(errno));
        unlink(dev_file_path);
        return status;
    }

    pFpgaHandleInt->device_file_path = strdup(dev_file_path);

    pFpgaHandleInt->input_buffer = (DWORD **) malloc(sizeof(DWORD*) * CATAPULT_FPGA_SLOT_NUM);
    pFpgaHandleInt->output_buffer = (DWORD **) malloc(sizeof(DWORD*) * CATAPULT_FPGA_SLOT_NUM);
    pFpgaHandleInt->result_buffer = (DWORD **) malloc(sizeof(DWORD*) * CATAPULT_FPGA_SLOT_NUM);

    if (pFpgaHandleInt->input_buffer == 0
        || pFpgaHandleInt->output_buffer == 0
        || pFpgaHandleInt->result_buffer == 0) {
        LOG_ERROR(status, FPGA_STATUS_ERROR_OPENING_DEVICE, "Failed to allocate memory for FPGA buffers");
        unlink(dev_file_path);
        return status;
    }

	for (int i = 0; i < CATAPULT_FPGA_SLOT_NUM; i++) {
		pFpgaHandleInt->input_buffer[i] = (DWORD *)((unsigned char *)input_buffer + CATAPULT_FPGA_DATA_BUF_SIZE_PER_SLOT * i);
		pFpgaHandleInt->output_buffer[i] = (DWORD *)((unsigned char *)output_buffer + CATAPULT_FPGA_DATA_BUF_SIZE_PER_SLOT * i);
		pFpgaHandleInt->result_buffer[i] = (DWORD *)((unsigned char *)result_buffer + CATAPULT_FPGA_META_BUF_SIZE_PER_SLOT * i);
    }

    return FPGA_STATUS_SUCCESS;
}

FPGA_STATUS _FPGA_CloseHandle(_In_ FPGA_HANDLE fpgaHandle, _In_ bool errorOnRootport= NULL)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHandleInt->device_file_path)
        unlink(fpgaHandleInt->device_file_path);

/*
    if (!InterlockedBitTestAndReset64(&handlesOpenedPerProcess, fpgaHandleInt->epnum))
    {
        LOG_ERROR(status, FPGA_STATUS_CANNOT_CLOSE_MULTIPLE_HANDLES, "Attempted to close a handle with an endpoint number not already opened by this process (ep:%u)", fpgaHandleInt->epnum);
        goto finish;
    }

    // Close rootport handle here, cannot move this further inside since handle needs to stay live during reconfig
    _FPGA_CloseRootportHandle(fpgaHandle, errorOnRootportNULL);

    // Full cleanup of handle objects, but not the handle itself.
    if ((status = _FPGA_Close(fpgaHandle)) != FPGA_STATUS_SUCCESS)
        goto finish;
    */

finish:
    // Free the handle itself.
    if (fpgaHandle != NULL)
    {
        free(fpgaHandle);
    }
    return status;
}


_CDLLEXPORT_ FPGA_STATUS FPGA_CreateHandle(_Out_ FPGA_HANDLE *fpgaHandle, _In_ DWORD endpointNumber, _In_ DWORD flags, _In_ const char *pchVerDefnsFile, _In_ const char *pchVerManifestFile, _In_ void(*logFunc)(DWORD logflag, const char *pchString))
{
	FPGA_HANDLE_INT *pFpgaHandleInt = NULL;
	FPGA_STATUS status = FPGA_STATUS_SUCCESS;

    // DO NOT CALL FPGA_IsDevicePresent here, because old clients may not have [PnP] section in INI file yet. Add this later.

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (endpointNumber >= 2)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_ENDPOINT_NUMBER, "Illegal endpoint number");
        goto finish;
    }

    /*
    if (InterlockedBitTestAndSet64(&handlesOpenedPerProcess, endpointNumber))
    {
        LOG_ERROR(status, FPGA_STATUS_CANNOT_OPEN_MULTIPLE_HANDLES, "Attempted to open more than one handle per process (ep:%u)", endpointNumber);
        goto finish;
    }
    */

    pFpgaHandleInt = (FPGA_HANDLE_INT *)malloc(sizeof(FPGA_HANDLE_INT));

    if (pFpgaHandleInt == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_MEMORY_NOT_ALLOCATED, "Unable to allocate memory for FPGA handle");
        goto finish;
    }

    memset(pFpgaHandleInt, 0x0, sizeof(FPGA_HANDLE_INT));
    *fpgaHandle = (FPGA_HANDLE)pFpgaHandleInt;

    pFpgaHandleInt->epnum = endpointNumber;
    pFpgaHandleInt->flags = flags;

    if (logFunc == NULL)
        pFpgaHandleInt->logfunc = _FPGA_DefaultLogger;
    else
        pFpgaHandleInt->logfunc = logFunc;

    if ((status = map_buffers(pFpgaHandleInt, endpointNumber)) != FPGA_STATUS_SUCCESS) {
        goto finish;
    }

    pFpgaHandleInt->num_pc_buffers = CATAPULT_FPGA_SLOT_NUM;
    pFpgaHandleInt->num_pc_buffers_raw = CATAPULT_FPGA_SLOT_NUM;
    pFpgaHandleInt->fpga_buffer_size_bytes = CATAPULT_FPGA_DATA_BUF_SIZE_PER_SLOT;
    pFpgaHandleInt->fpga_buffer_size_raw = CATAPULT_FPGA_DATA_BUF_SIZE_PER_SLOT;
    pFpgaHandleInt->num_general_purpose_registers = CATAPULT_FPGA_NUM_GENERAL_PURPOSE_REGISTERS;
    pFpgaHandleInt->input_buffer_bytes = CATAPULT_FPGA_DATA_BUF_SIZE;
    pFpgaHandleInt->output_buffer_bytes = CATAPULT_FPGA_META_BUF_SIZE;

	if ((status = _FPGA_GetSoftRegisterCapability(*fpgaHandle)) != FPGA_STATUS_SUCCESS) {
		goto finish;
	}

finish: 
    if (status != FPGA_STATUS_SUCCESS)
    {
        if (pFpgaHandleInt != NULL)
        {
            _FPGA_CloseHandle((FPGA_HANDLE)pFpgaHandleInt, pFpgaHandleInt->rootport_handle != NULL);
        }
        if (fpgaHandle != NULL)
        {
            *fpgaHandle = NULL;
        }
    }
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetNumberEndpoints(_In_ FPGA_HANDLE fpgaHandle, _Out_ DWORD *numEndpoints)
{
    DWORD status = FPGA_STATUS_SUCCESS;
    LOG_ERROR(status, FPGA_STATUS_UNIMPLEMENTED, "Unsupported API call");
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetNumberBuffers(_In_ FPGA_HANDLE fpgaHandle, _Out_ DWORD *numBuffers)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;
	if (fpgaHandle == NULL || numBuffers == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or numBuffers");
        goto finish;
    }
    *numBuffers = fpgaHandleInt->num_pc_buffers;

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetBufferSize(_In_ FPGA_HANDLE fpgaHandle, _Out_ DWORD *bufferBytes)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;

    *bufferBytes = 0;
	if (fpgaHandle == NULL || bufferBytes == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or bufferBytes");
        goto finish;
    }
	*bufferBytes = fpgaHandleInt->fpga_buffer_size_bytes;

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetInputBufferPointer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ PDWORD *inputBufferPtr)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;

    *inputBufferPtr = NULL;
	if (fpgaHandle == NULL || inputBufferPtr == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or input buffer");
		goto finish;
    }
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer: %u, num_pc_buffer:%u", whichBuffer, fpgaHandleInt->num_pc_buffers);
        goto finish;
	}
    *inputBufferPtr = (PDWORD)fpgaHandleInt->input_buffer[whichBuffer];

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetOutputBufferPointer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ PDWORD *outputBufferPtr)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;
	if (fpgaHandle == NULL || outputBufferPtr == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or output buffer");
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer: %u, num_pc_buffer:%u", whichBuffer, fpgaHandleInt->num_pc_buffers);
        goto finish;
	}
    *outputBufferPtr = (PDWORD)fpgaHandleInt->output_buffer[whichBuffer];

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetResultBufferPointer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ PDWORD *resultBufferPtr)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD status = FPGA_STATUS_SUCCESS;

	if (fpgaHandle == NULL || resultBufferPtr == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or result buffer");
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer: %u, num_pc_buffer:%u", whichBuffer, fpgaHandleInt->num_pc_buffers);
        goto finish;
	}

    *resultBufferPtr = (PDWORD)fpgaHandleInt->result_buffer[whichBuffer];

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_ReadSoftRegister(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD address, _Out_ DWORD64 *readValue)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    DWORD rdval, rdval2;
    DWORD waitRet;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }
    if ((address >> 16) != 0)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_REGISTER_NUMBER, "Error, soft register addresses are only valid between 0-65535, got:%u", address);
        goto finish;
    }
    if (fpgaHandleInt->soft_registers_native_64_support)	//native 64-bit interface
    {
		*readValue = _FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, address);
	}
	else
	{
        /*
		waitRet = WaitForSingleObject(fpgaHandleInt->mutex_softreg, MUTEX_TIMEOUT_MILLISECONDS);

		if (waitRet == WAIT_TIMEOUT)
		{
			LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Soft register access timed out after %u milliseconds", MUTEX_TIMEOUT_MILLISECONDS);
			goto finish;
		}

		if ((waitRet != WAIT_OBJECT_0) && (waitRet != WAIT_ABANDONED))
		{
			LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Soft register access failed during mutex wait, GetLastError:%u", GetLastError());
			goto finish;
		}
        */

		if ((status = FPGA_WriteShellRegister(fpgaHandle, 12, address)) != FPGA_STATUS_SUCCESS)
			goto finish;

		if ((status = FPGA_WriteShellRegister(fpgaHandle, 15, 0x0)) != FPGA_STATUS_SUCCESS)
			goto finish;

		if ((status = FPGA_ReadShellRegister(fpgaHandle, 13, &rdval)) != FPGA_STATUS_SUCCESS) // obtain lower 32 bits of return value
			goto finish;

		if ((status = FPGA_ReadShellRegister(fpgaHandle, 14, &rdval2)) != FPGA_STATUS_SUCCESS)
			goto finish;

		*readValue = (DWORD64)rdval;
		*readValue |= ((DWORD64)rdval2 << 32ULL); // obtain upper 32 bits of return value

        /*
		if (ReleaseMutex(fpgaHandleInt->mutex_softreg) == 0)
		{
			LOG_ERROR(status, FPGA_STATUS_MUTEX_ERROR, "Soft register mutex release failed, GetLastError:%u", GetLastError());
			goto finish;
		}
        */
	}

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_WriteSoftRegister(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD registerNumber, _In_ DWORD64 writeValue)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }
    if ((registerNumber >> 16) != 0)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_REGISTER_NUMBER, "Error, soft register addresses are only valid between 0-65535, got:%u", registerNumber);
        goto finish;
    }
    if (fpgaHandleInt->soft_registers_native_64_support)	//native 64-bit interface
    {
		_FPGA_low_level_write_64(fpgaHandle, INTER_ADDR_SOFT_REG, registerNumber, writeValue);
    }
	else 	//legacy 32-bit interface
	{
        /*
		DWORD waitRet = WaitForSingleObject(fpgaHandleInt->mutex_softreg, MUTEX_TIMEOUT_MILLISECONDS);

		if (waitRet == WAIT_TIMEOUT)
		{
			LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Soft register access timed out after %u milliseconds", MUTEX_TIMEOUT_MILLISECONDS);
			goto finish;
		}

		if ((waitRet != WAIT_OBJECT_0) && (waitRet != WAIT_ABANDONED))
		{
			LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Soft register access failed during mutex wait, GetLastError:%u", GetLastError());
			goto finish;
		}
        */

		if ((status = FPGA_WriteShellRegister(fpgaHandle, 12, registerNumber)) != FPGA_STATUS_SUCCESS) // arm address
			goto finish;

		if ((status = FPGA_WriteShellRegister(fpgaHandle, 13, (DWORD)(writeValue & 0xFFFFFFFFU))) != FPGA_STATUS_SUCCESS) // store lower 32 bits of write value
			goto finish;

		if ((status = FPGA_WriteShellRegister(fpgaHandle, 14, (DWORD)((writeValue >> 32) & 0xFFFFFFFFU))) != FPGA_STATUS_SUCCESS) // store upper 32 bits of write value
			goto finish;

        /*
		if (ReleaseMutex(fpgaHandleInt->mutex_softreg) == 0)
		{
			LOG_ERROR(status, FPGA_STATUS_MUTEX_ERROR, "Soft register mutex release failed, GetLastError:%u", GetLastError());
			goto finish;
		}
        */
	}

finish:
	return status;
}


_CDLLEXPORT_ FPGA_STATUS FPGA_GetInputBufferFull(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ BOOL *isFull)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

	if (fpgaHandle == NULL || isFull == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or isFull");
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer");
        goto finish;
	}

	*isFull = (fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 1] == 1);

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetOutputBufferDone(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ BOOL *isDone)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
	if (fpgaHandle == NULL || isDone == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or isDone");
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer: %u", whichBuffer);
        goto finish;
	}

	*isDone = fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3];

finish:
	return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_SendInputBuffer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _In_ DWORD sizeBytes, _In_ BOOL useInterrupt)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD num_128_bit_words;
    
    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }
	if (fpgaHandleInt->flags & FPGA_HANDLE_FLAG_DIAGNOSTICS)
	{
		LOG_ERROR(status, FPGA_STATUS_CANNOT_USE_WITH_DIAGNOSTICS, "DMA is not available when FPGA handle is opened in diagnostic mode");
		goto finish;
	}
    if ((sizeBytes % 16) != 0)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_SIZE, "Error, size in bytes must be 16B-aligned, got:%u", sizeBytes);
        goto finish;
    }
    if (sizeBytes == 0)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_SIZE, "Must provide non-zero size, got:%u", sizeBytes);
        goto finish;
    }
    if (sizeBytes < 32)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_SIZE, "Size must be 32B or greater, got:%u\n", sizeBytes);
        goto finish;
    }
    if (sizeBytes > fpgaHandleInt->fpga_buffer_size_bytes)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_SIZE, "Exceeded maximum allowed send size, got:%u", sizeBytes);
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer, got:%u", whichBuffer);
        goto finish;
	}

	num_128_bit_words = sizeBytes >> 4;
	fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 1] = 1;
	_FPGA_low_level_write(fpgaHandle, INTER_ADDR_FULL_STATUS_REG, whichBuffer, (useInterrupt) ? ((1 << 31) | num_128_bit_words) : num_128_bit_words);

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_WaitOutputBuffer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ DWORD *pBytesReceived, _In_ BOOL useInterrupt, _In_ double timeoutInSeconds)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD retBytes;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Error, null pointer given for FPGA handle");
        goto finish;
	}
	if (fpgaHandleInt->flags & FPGA_HANDLE_FLAG_DIAGNOSTICS)
	{
		LOG_ERROR(status, FPGA_STATUS_CANNOT_USE_WITH_DIAGNOSTICS, "DMA is not available when FPGA handle is opened in diagnostic mode");
		goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer, got:%u", whichBuffer);
        goto finish;
	}

	if ((status = _FPGA_wait_for_done(fpgaHandle, whichBuffer, &retBytes, useInterrupt, timeoutInSeconds)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if (retBytes < 32)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA returned less than 32B, got:%u", retBytes);
        goto finish;
    }
    if ((retBytes % 16) != 0)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA return non-16B-aligned size, got:%u", retBytes);
        goto finish;
    }
    if (retBytes == 0)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA returned zero size, got:%u\n", retBytes);
        goto finish;
    }
    if ((retBytes - 16) > fpgaHandleInt->fpga_buffer_size_bytes)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, exceeded maximum return size, got:%u\n", retBytes);
        goto finish;
    }
    if (pBytesReceived != NULL)
    {
        *pBytesReceived = retBytes;
    }

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_WaitAnyOutputBuffer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD startBuffer, _In_ DWORD endBuffer, _In_ DWORD first, _Out_ DWORD* whichBuffer, _Out_ DWORD *pBytesReceived, _In_ BOOL useInterrupt, _In_ double timeoutInSeconds)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD retBytes;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Error, null pointer given for FPGA handle");
        goto finish;
    }
    if (whichBuffer == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Error, null pointer given for whichBuffer");
        goto finish;
    }
    if (fpgaHandleInt->flags & FPGA_HANDLE_FLAG_DIAGNOSTICS)
    {
        LOG_ERROR(status, FPGA_STATUS_CANNOT_USE_WITH_DIAGNOSTICS, "DMA is not available when FPGA handle is opened in diagnostic mode");
        goto finish;
    }

    if ((status = _FPGA_wait_for_any_done(fpgaHandle, startBuffer, endBuffer, first, whichBuffer, &retBytes, useInterrupt, timeoutInSeconds)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if (retBytes < 32)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA returned less than 32B, got:%u", retBytes);
        goto finish;
    }
    if ((retBytes % 16) != 0)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA return non-16B-aligned size, got:%u", retBytes);
        goto finish;
    }
    if (retBytes == 0)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, FPGA returned zero size, got:%u\n", retBytes);
        goto finish;
    }
    if ((retBytes - 16) > fpgaHandleInt->fpga_buffer_size_bytes)
    {
        LOG_ERROR(status, FPGA_STATUS_BAD_SIZE_RETURNED, "Error, exceeded maximum return size, got:%u\n", retBytes);
        goto finish;
    }
    if (pBytesReceived != NULL)
    {
        *pBytesReceived = retBytes;
    }

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_DiscardOutputBuffer(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status,FPGA_STATUS_NULL_POINTER_GIVEN, "Error, null pointer given for FPGA handle\n");
        goto finish;
	}
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
	{
        LOG_ERROR(status,FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Invalid value for whichBuffer, got:%u", whichBuffer);
        goto finish;
	}

	if ((status = _FPGA_clear_done_status(fpgaHandle, whichBuffer)) != FPGA_STATUS_SUCCESS)
        goto finish;

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_CloseHandle(_In_ FPGA_HANDLE fpgaHandle)
{
    return _FPGA_CloseHandle(fpgaHandle);
}

/*
_CDLLEXPORT_ VOID FPGA_kvs_write_256(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD64 *writeData)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	//printf("Log: Trigger 256-bit low level write, bar0_registers_kvs = %x", fpgaHandleInt -> bar0_registers_kvs);
	HW_register_write256((DWORD64*)(fpgaHandleInt->bar0_registers_kvs), writeData);

}

__declspec(align(16)) DWORD64 aligned_data[2];

_CDLLEXPORT_ VOID FPGA_kvs_write_128(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD64 *writeData)
{
        memcpy(aligned_data, writeData, 16);
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	//printf("Log: Trigger 128-bit low level write, bar0_registers_kvs = %x", fpgaHandleInt -> bar0_registers_kvs);
	HW_register_write128((DWORD64*)(fpgaHandleInt->bar0_registers_kvs), aligned_data);

}
*/
