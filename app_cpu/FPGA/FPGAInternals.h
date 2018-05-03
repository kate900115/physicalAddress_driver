#ifndef __FPGAInternals__
#define __FPGAInternals__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "FPGACoreLib.h"
#include "FPGACore.h"
#include "FPGADriverAPI.h"
#define _CDLLEXPORT_

extern char __errMessage[MAX_BYTES_LOG_ERROR];
extern FPGA_STATUS __lastError;

DWORD _FPGA_low_level_read_legacy(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	ULONG_PTR byte_address = (appAddress << 8) | (interpAddress << 4) | 4;
	return HW_register_read32((DWORD*)(fpgaHandleInt->bar0_registers + byte_address));		//implemented in DLL.cpp, no error code returned
}

VOID _FPGA_low_level_write_legacy(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress, _In_ DWORD writeData)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	ULONG_PTR byte_address = (appAddress << 8) | (interpAddress << 4) | 4;
	HW_register_write32((DWORD*)(fpgaHandleInt->bar0_registers + byte_address), writeData);	//implemented in DLL.cpp, no error code returned
}

DWORD64 _FPGA_low_level_read_64(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	ULONG_PTR byte_address = (appAddress << 8) | (interpAddress << 4);
	return HW_register_read64((DWORD64*)(fpgaHandleInt->bar0_registers + byte_address));
}

VOID _FPGA_low_level_write_64(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress, _In_ DWORD64 writeData)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	ULONG_PTR byte_address = (appAddress << 8) | (interpAddress << 4);
	HW_register_write64((DWORD64*)(fpgaHandleInt->bar0_registers + byte_address), writeData);
}

DWORD _FPGA_low_level_read(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	DWORD readData;
	if (fpgaHandleInt->soft_registers_mapping_slot_dma) {
		switch (interpAddress & 0xf) {
		case INTER_ADDR_FULL_STATUS_REG:	//instead of 64 addresses each 1 bit, now it is 1 address with 64 bits, unpack results in software
			readData = (DWORD)((_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 62) >> appAddress) & 1);
			break;
		case INTER_ADDR_DONE_STATUS_REG:
			readData = (DWORD)((_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 61) >> appAddress) & 1);
			break;
		case INTER_ADDR_PEND_STATUS_REG:
			readData = (DWORD)((_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 60) >> appAddress) & 1);
			break;
		case INTER_ADDR_GENERAL_PURPOSE_REG:
			readData = _FPGA_low_level_read_legacy(fpgaHandle, interpAddress, appAddress);
			break;
		case INTER_ADDR_ASMI_RSU:
			readData = _FPGA_low_level_read_legacy(fpgaHandle, interpAddress, appAddress);
			break;
		case INTER_ADDR_HACK_OVERRIDE_OUT_DATA_SIZE:
			if (appAddress >= 2 && appAddress <= 6) {
				readData = (DWORD)_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 55 + (appAddress - 2));
			}
			else readData = 0;
			break;
		case INTER_ADDR_INTERRUPT:
			if (appAddress == 257) {
				readData = (DWORD)_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 54);
			}
			else readData = 0;
			break;
		case INTER_ADDR_DMA_DESCRIPTORS_AND_RESERVED:
			if (appAddress <= 53) {
				if (appAddress == 4 || appAddress == 5 || appAddress == 6)	//force legacy, even if we have soft reg capability, role may not have these registers
					readData = _FPGA_low_level_read_legacy(fpgaHandle, interpAddress, appAddress);
				else	//0-3, 7-53 mapping for the factory tester registers
					readData = (DWORD)_FPGA_low_level_read_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + appAddress);
			}
			else readData = 0;
			break;
		default:
			readData = 0;
		}
	}
	else {
		readData = _FPGA_low_level_read_legacy(fpgaHandle, interpAddress, appAddress);
	}
//	printf("low level read, hip %x, interp %x, app %x, data %x\n", fpgaHandleInt->epnum, interpAddress, appAddress, readData);
	return readData;
}


VOID _FPGA_low_level_write(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD interpAddress, _In_ DWORD appAddress, _In_ DWORD writeData)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
//	printf("low level write, hip %x, interp %x, app %x, data %x\n", fpgaHandleInt->epnum, interpAddress, appAddress, writeData);
	if (fpgaHandleInt->soft_registers_mapping_slot_dma) {
		switch (interpAddress & 0xf) {
		case INTER_ADDR_GENERAL_PURPOSE_REG:
			_FPGA_low_level_write_legacy(fpgaHandle, interpAddress, appAddress, writeData);
			break;
		case INTER_ADDR_ASMI_RSU:
			_FPGA_low_level_write_legacy(fpgaHandle, interpAddress, appAddress, writeData);
			break;
		default:
			DWORD64 wrData = (appAddress << 8) | (interpAddress << 4) | 4;
			wrData = (wrData << 32) | writeData;
			_FPGA_low_level_write_64(fpgaHandle, INTER_ADDR_SOFT_REG, SOFT_REG_SLOT_DMA_BASE_ADDR + 63, wrData);
			break;
		}
	}
	else {
		_FPGA_low_level_write_legacy(fpgaHandle, interpAddress, appAddress, writeData);
	}
}


FPGA_STATUS _FPGA_clear_full_status(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
    {
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Illegal buffer number: %u", whichBuffer);
        goto finish;
    }

	fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 1] = 0;

finish:
    return status;
}

FPGA_STATUS _FPGA_clear_done_status(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
    {
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Illegal buffer number: %u", whichBuffer);
        goto finish;
    }

    //ResetEvent(fpgaHandleInt->isr_wake_up_events[whichBuffer]);
    fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3] = 0;
	_FPGA_low_level_write(fpgaHandle, INTER_ADDR_DONE_STATUS_REG, whichBuffer, 0);
	
finish:
    return status;
}

FPGA_STATUS _FPGA_wait_for_done(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD whichBuffer, _Out_ DWORD *retBytes, _In_ BOOL useInterrupt, _In_ double timeoutInSeconds)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    volatile DWORD tmp;

	if (whichBuffer >= fpgaHandleInt->num_pc_buffers)
    {
		LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Illegal buffer number: %u", whichBuffer);
        goto finish;
    }

    //
    // Option 1: Interrupt with timeout.
    //
    if ((useInterrupt == TRUE) && (timeoutInSeconds != 0.0))
    {
        LOG_ERROR(status, FPGA_STATUS_ERROR_ENABLING_INTERRUPTS, "Interrupt not supported");
        /*
		DWORD waitStatus = WaitForSingleObject(fpgaHandleInt->isr_wake_up_events[whichBuffer], (DWORD)(timeoutInSeconds * 1000.0));
        if (waitStatus == WAIT_TIMEOUT)
        {
			LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Wait on interrupt event timed out while waiting for done status on output buffer %u", whichBuffer);
            goto finish;
        }
        else if (waitStatus != WAIT_OBJECT_0)
        {
			LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Wait on interrupt event failed (status:%u) while waiting for done status on output buffer %u", waitStatus, whichBuffer);
            goto finish;
        }
		ResetEvent(fpgaHandleInt->isr_wake_up_events[whichBuffer]);	// after being woken by event, set the event back to non-active state
        */
    }
    //
    // Option 2: Interrupt with no timeout.
    //
    else if ((useInterrupt == TRUE) && (timeoutInSeconds == 0.0))
    {
        LOG_ERROR(status, FPGA_STATUS_ERROR_ENABLING_INTERRUPTS, "Interrupt not supported");
        /*
		DWORD waitStatus = WaitForSingleObject(fpgaHandleInt->isr_wake_up_events[whichBuffer], INFINITE);
        if (waitStatus != WAIT_OBJECT_0)
        {
			LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Wait on interrupt event failed (status:%u) while waiting for done status on output buffer %u", waitStatus, whichBuffer);
            goto finish;
        }
		ResetEvent(fpgaHandleInt->isr_wake_up_events[whichBuffer]);	// after being woken by event, set the event back to non-active state
        */
    }
    //
    // Option 3: Polling with no timeout.
    //
    if ((useInterrupt == FALSE) && (timeoutInSeconds == 0.0))
    {
        do {
			tmp = fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3];
        } while (tmp == 0);
    }
    //
    // Option 4: Polling with timeout.
    //
    else if ((useInterrupt == FALSE) && (timeoutInSeconds != 0.0))
    {
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        double timestamp_end = (double)ts.tv_sec + ts.tv_nsec / 1e9;
        do {
			tmp = fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3];

            timespec_get(&ts, TIME_UTC);
            double timestamp_now = (double)ts.tv_sec + ts.tv_nsec / 1e9;
            if (timestamp_now > timestamp_end)
                break;
        } while (tmp == 0);

        if (tmp == 0)
        {
			LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Timed out for % u seconds while polling on done status of output buffer %u", timeoutInSeconds, whichBuffer);
            goto finish;
        }
    }

	*retBytes = fpgaHandleInt->result_buffer[whichBuffer][0];

finish:
    return status;
}

FPGA_STATUS _FPGA_wait_for_any_done(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD startBuffer, _In_ DWORD endBuffer, _In_ DWORD first, _Out_ DWORD *outBuffer, _Out_ DWORD *retBytes, _In_ BOOL useInterrupt, _In_ double timeoutInSeconds)
{
    FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    volatile DWORD tmp;

    DWORD whichBuffer = startBuffer;

    if (startBuffer >= fpgaHandleInt->num_pc_buffers)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Illegal buffer number: %u", startBuffer);
        goto finish;
    }

    if (endBuffer >= fpgaHandleInt->num_pc_buffers)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_NUMBER, "Illegal buffer number: %u", endBuffer);
        goto finish;
    }

    if (endBuffer < startBuffer)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_RANGE, "Illegal buffer range: %u-%u", startBuffer, endBuffer);
        goto finish;
    }

    if (first < startBuffer || first > endBuffer)
    {
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_BUFFER_RANGE, "first buffer to check must be in buffer range. $u not in %u-%u.", first, startBuffer, endBuffer);
        goto finish;
    }

    /*
    DWORD nHandles = endBuffer - startBuffer + 1;
    if ((useInterrupt == TRUE) && (nHandles > MAXIMUM_WAIT_OBJECTS))
    {
        LOG_ERROR(status, FPGA_STATUS_EXCEEDED_MAXIMUM_BUFFER_RANGE, "Cannot wait on more than %u buffers while using interrupts.", MAXIMUM_WAIT_OBJECTS);
        goto finish;
    }
    */

    //
    // Option 1: Interrupt with timeout.
    //
    if ((useInterrupt == TRUE) && (timeoutInSeconds != 0.0))
    {
        LOG_ERROR(status, FPGA_STATUS_ERROR_ENABLING_INTERRUPTS, "Interrupt not supported");
        /*
        HANDLE events[MAXIMUM_WAIT_OBJECTS];
        DWORD srcIndex = first;
        for (DWORD destIndex = 0; destIndex < nHandles; destIndex++)
        {
            events[destIndex] = fpgaHandleInt->isr_wake_up_events[srcIndex];
            srcIndex++;
            if (srcIndex > endBuffer)
                srcIndex = startBuffer;
        }

        DWORD waitStatus = WaitForMultipleObjects(nHandles, events, FALSE, (DWORD)(timeoutInSeconds * 1000.0));
        if (waitStatus == WAIT_TIMEOUT)
        {
            LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Wait on interrupt event timed out while waiting for done status on output buffer range %u-%u", startBuffer, endBuffer);
            goto finish;
        }
        else if (waitStatus < WAIT_OBJECT_0 || waitStatus > WAIT_OBJECT_0 + nHandles - 1)
        {
            LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Wait on interrupt event failed (status:%u) while waiting for done status on output buffer range %u-%u", waitStatus, startBuffer, endBuffer);
            goto finish;
        }
        whichBuffer = waitStatus - WAIT_OBJECT_0;
        //figure out which slot was selected
        if (whichBuffer > endBuffer - first)
            whichBuffer = startBuffer + (whichBuffer - (endBuffer - first + 1));
        else
            whichBuffer = first + whichBuffer;

        ResetEvent(fpgaHandleInt->isr_wake_up_events[whichBuffer]);	// after being woken by event, set the event back to non-active state
        */
    }
    //
    // Option 2: Interrupt with no timeout.
    //
    else if ((useInterrupt == TRUE) && (timeoutInSeconds == 0.0))
    {
        LOG_ERROR(status, FPGA_STATUS_ERROR_ENABLING_INTERRUPTS, "Interrupt not supported");
        /*
        HANDLE events[MAXIMUM_WAIT_OBJECTS];
        DWORD srcIndex = first;
        for (DWORD destIndex = 0; destIndex < nHandles; destIndex++)
        {
            events[destIndex] = fpgaHandleInt->isr_wake_up_events[srcIndex];
            srcIndex++;
            if (srcIndex > endBuffer)
                srcIndex = startBuffer;
        }

        DWORD waitStatus = WaitForMultipleObjects(nHandles, events, FALSE, INFINITE);
        if (waitStatus < WAIT_OBJECT_0 || waitStatus > WAIT_OBJECT_0 + nHandles - 1)
        {
            LOG_ERROR(status, FPGA_STATUS_WAIT_FAILED, "Wait on interrupt event failed (status:%u) while waiting for done status on output buffer range %u-%u", waitStatus, startBuffer, endBuffer);
            goto finish;
        }
        whichBuffer = waitStatus - WAIT_OBJECT_0;
        //figure out which slot was selected
        if (whichBuffer > endBuffer - first)
            whichBuffer = startBuffer + (whichBuffer - (endBuffer - first + 1));
        else
            whichBuffer = first + whichBuffer;

        ResetEvent(fpgaHandleInt->isr_wake_up_events[whichBuffer]);	// after being woken by event, set the event back to non-active state
        */
    }
    //
    // Option 3: Polling with no timeout.
    //
    if ((useInterrupt == FALSE) && (timeoutInSeconds == 0.0))
    {
        if (first == startBuffer)
            whichBuffer = endBuffer;
        else
            whichBuffer = first - 1;
        do {
            if (whichBuffer == endBuffer)
                whichBuffer = startBuffer;
            else
                whichBuffer++;

            tmp = fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3];
        } while (tmp == 0);
    }
    //
    // Option 4: Polling with timeout.
    //
    else if ((useInterrupt == FALSE) && (timeoutInSeconds != 0.0))
    {
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        double timestamp_end = (double)ts.tv_sec + ts.tv_nsec / 1e9;

        if (first == startBuffer)
            whichBuffer = endBuffer;
        else
            whichBuffer = first - 1;
        do {
            if (whichBuffer == endBuffer)
                whichBuffer = startBuffer;
            else
                whichBuffer++;

            tmp = fpgaHandleInt->ctrl_buffer[16 * whichBuffer + 3];

            timespec_get(&ts, TIME_UTC);
            double timestamp_now = (double)ts.tv_sec + ts.tv_nsec / 1e9;
            if (timestamp_now > timestamp_end)
                break;
        } while (tmp == 0);

        if (tmp == 0)
        {
            LOG_ERROR(status, FPGA_STATUS_WAIT_TIMEOUT, "Timed out for % u seconds while polling on done status of output buffer range %u-%u", timeoutInSeconds, startBuffer, endBuffer);
            goto finish;
        }
    }

    *retBytes = fpgaHandleInt->result_buffer[whichBuffer][0];

finish:
    *outBuffer = whichBuffer;
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_ReadShellRegister(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD registerNumber, _Out_ DWORD *readValue)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

	if (fpgaHandle == NULL || readValue == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or readValue");
        goto finish;
	}
	if (registerNumber >= fpgaHandleInt->num_general_purpose_registers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_REGISTER_NUMBER, "Invalid value for registerNumber, got:%u (total %u registers)", registerNumber, fpgaHandleInt->num_general_purpose_registers);
        goto finish;
	}

	*readValue = _FPGA_low_level_read(fpgaHandle, INTER_ADDR_GENERAL_PURPOSE_REG, registerNumber);

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_WriteShellRegister(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD registerNumber, _In_ DWORD writeValue)
{
	FPGA_HANDLE_INT *fpgaHandleInt = (FPGA_HANDLE_INT *)fpgaHandle;
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle or writeValue");
        goto finish;
    }
	if (registerNumber >= fpgaHandleInt->num_general_purpose_registers)
	{
        LOG_ERROR(status, FPGA_STATUS_ILLEGAL_REGISTER_NUMBER, "Invalid value for registerNumber, got:%u", registerNumber);
        goto finish;
	}

	_FPGA_low_level_write(fpgaHandle, INTER_ADDR_GENERAL_PURPOSE_REG, registerNumber, writeValue);

finish:
    return status;
}

#endif
