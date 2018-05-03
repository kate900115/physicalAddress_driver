#ifndef __FPGAManagementLib__
#define __FPGAManagementLib__

#define FPGA_STAT_NIC_TX_PACKETS                    0
#define FPGA_STAT_NIC_RX_PACKETS                    1
#define FPGA_STAT_NIC_RX_FCS_ERRORS                 2
#define FPGA_STAT_NIC_LINK_DROPS                    3
            
#define FPGA_STAT_TOR_TX_PACKETS                    4 
#define FPGA_STAT_TOR_RX_PACKETS                    5
#define FPGA_STAT_TOR_RX_FCS_ERRORS                 6
#define FPGA_STAT_TOR_LINK_DROPS                    7
                        
#define FPGA_STAT_CYCLES_LOWER                      8
#define FPGA_STAT_CYCLES_UPPER                      9
#define FPGA_STAT_CRC_ERRORS                        10 
#define FPGA_STAT_DRAM_0_SINGLE_ERRORS              11
#define FPGA_STAT_DRAM_0_DOUBLE_ERRORS              12
#define FPGA_STAT_DRAM_0_UNCORRECTED_ERRORS         13

_CDLLEXPORT_ FPGA_STATUS FPGA_IsGoldenImage(_In_ FPGA_HANDLE fpgaHandle, _Out_ BOOL *isGolden, _Out_ DWORD *roleIDOut, _Out_ DWORD *roleVersionOut)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD roleVersion;
    DWORD roleID;

    const int GOLDEN_IMAGE_VERSION_1         = 0xFAC00100;
	const int GOLDEN_IMAGE_VERSION_2         = 0xbfb20002;
	const int GOLDEN_IMAGE_VERSION_3         = 0x00000000;
	const int GOLDEN_IMAGE_VERSION_4_PREFIX  = 0xCA7A0000; //Pikes Peak Golden Images - last four digits are Revision Numbers MMmm
	const int GOLDEN_IMAGE_VERSION_5_PREFIX  = 0xCA7B0000; //Dragon Tail Peak

    const int GOLDEN_ROLE_ID_1               = 0x0;
    const int GOLDEN_ROLE_ID_2               = 0x0;
    const int GOLDEN_ROLE_ID_3               = 0x0;
    const int GOLDEN_ROLE_ID_4               = 0x0;
    const int GOLDEN_ROLE_ID_4PLUS           = 0x601D;
    const int GOLDEN_ROLE_ID_5PLUS           = 0x601D;

    if (!fpgaHandle)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 65, &roleVersion)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 101, &roleID)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if (isGolden)
    {
        *isGolden = ((roleVersion == GOLDEN_IMAGE_VERSION_1) && (roleID == GOLDEN_ROLE_ID_1)) || 
                    ((roleVersion == GOLDEN_IMAGE_VERSION_2) && (roleID == GOLDEN_ROLE_ID_2)) || 
                    ((roleVersion == GOLDEN_IMAGE_VERSION_3) && (roleID == GOLDEN_ROLE_ID_3)) || 
                    (((roleVersion & 0xFFFF0000) == GOLDEN_IMAGE_VERSION_4_PREFIX) && (roleID == GOLDEN_ROLE_ID_4)) ||
                    (((roleVersion & 0xFFFF0000) == GOLDEN_IMAGE_VERSION_4_PREFIX) && (roleID == GOLDEN_ROLE_ID_4PLUS)) ||
                    (((roleVersion & 0xFFFF0000) == GOLDEN_IMAGE_VERSION_5_PREFIX) && (roleID == GOLDEN_ROLE_ID_5PLUS));
    }

    if (roleVersionOut)
    {
        *roleVersionOut = roleVersion;
    }
    if (roleIDOut)
    {
        *roleIDOut = roleID;
    }
finish:
    return status;
}


_CDLLEXPORT_ FPGA_STATUS FPGA_SetNetworkBypass(_In_ FPGA_HANDLE fpgaHandle, _In_ BOOL forceBypass)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD ctrl_register, ctrl_checkval;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for handle");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 0x0, &ctrl_register)) != FPGA_STATUS_SUCCESS)
        goto finish;

    ctrl_register &= ~(1U << 4);
    ctrl_register |= (DWORD)(forceBypass) << 4;

    if ((status = FPGA_WriteShellRegister(fpgaHandle, 0x0, ctrl_register)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 0x0, &ctrl_checkval)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if (ctrl_register != ctrl_checkval)
    {
        LOG_ERROR(status, FPGA_STATUS_FAILED_WRITE, "Unable to set network bypass mode, got:0x%x, expect:0x%x", ctrl_checkval, ctrl_register);
        goto finish;
    }

finish:
    return status;
}


_CDLLEXPORT_ FPGA_STATUS FPGA_ReadStatistic(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD statType, _Out_ DWORD64 *value)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD shellNum = 0;
    DWORD rdval;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (value == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for storage");
        goto finish;
    }

    switch (statType)
    {
        case FPGA_STAT_NIC_TX_PACKETS: shellNum = 23; break;
        case FPGA_STAT_NIC_RX_PACKETS: shellNum = 24; break;
        case FPGA_STAT_NIC_RX_FCS_ERRORS: shellNum = 25; break;
        case FPGA_STAT_NIC_LINK_DROPS: shellNum = 26; break;
                    
        case FPGA_STAT_TOR_TX_PACKETS: shellNum = 19; break;
        case FPGA_STAT_TOR_RX_PACKETS: shellNum = 20; break;
        case FPGA_STAT_TOR_RX_FCS_ERRORS: shellNum = 21; break;
        case FPGA_STAT_TOR_LINK_DROPS: shellNum = 22; break;
                
        case FPGA_STAT_CYCLES_LOWER: shellNum = 66; break;
        case FPGA_STAT_CYCLES_UPPER: shellNum = 67; break;
        case FPGA_STAT_CRC_ERRORS: shellNum = 104; break;
        case FPGA_STAT_DRAM_0_SINGLE_ERRORS:
        case FPGA_STAT_DRAM_0_DOUBLE_ERRORS:
        case FPGA_STAT_DRAM_0_UNCORRECTED_ERRORS: shellNum = 74; break;
        default:
            LOG_ERROR(status, FPGA_STATUS_BAD_ARGUMENT, "Invalid argument when retrieving statistic");
            goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, shellNum, &rdval)) != FPGA_STATUS_SUCCESS)
        goto finish;

    *value = (DWORD64)rdval;

    if (statType == FPGA_STAT_DRAM_0_SINGLE_ERRORS)
        *value &= 0xFFFFULL;
    else if (statType == FPGA_STAT_DRAM_0_DOUBLE_ERRORS)
        *value = (*value >> 16) & 0xFFULL;
    else if (statType == FPGA_STAT_DRAM_0_UNCORRECTED_ERRORS)
        *value = (*value >> 24) & 0xFFULL;

finish:
    return status;
}

#endif
