#include "FPGACoreLib.h"
#include "FPGAHealthLib.h"
#include "FPGACore.h"
#include "FPGAManagementLib.h"

#define  _CDLLEXPORT_

extern char __errMessage[MAX_BYTES_LOG_ERROR];
extern FPGA_STATUS __lastError;

FPGA_STATUS _FPGA_GetHealthProperty_Temperature(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD temp_info;
    DWORD temp_warn, temp_shutdown, temp_c, min_temp_c, max_temp_c;
    char temp_buf[1024];

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 71, &temp_info)) != FPGA_STATUS_SUCCESS)
        goto finish;

    temp_warn = temp_info & 0x1;
    temp_shutdown = (temp_info >> 1) & 0x1;
    temp_c = (temp_info >> 8) & 0xFF;
    min_temp_c = (temp_info >> 16) & 0xFF;
    max_temp_c = (temp_info >> 24) & 0xFF;

    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_TEMPERATURE;
    sprintf_s(fpgaHealthProperty->name, "FPGA-TEMP");

    if (temp_shutdown)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT;
        sprintf_s(fpgaHealthProperty->message, "ERROR: Emergency shutdown detected due to unsafe temperature");
    }
    else if (temp_warn)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: High temperature detected");
    }
    else
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK");
    }

    sprintf_s(temp_buf, "[temp_warn:%u,temp_shutdown:%u,temp_c:%u,min_temp_c:%u,max_temp_c:%u]", temp_warn, temp_shutdown, temp_c, min_temp_c, max_temp_c);
    strcat_s(fpgaHealthProperty->message, " ");
    strcat_s(fpgaHealthProperty->message, temp_buf);

finish:
    return status;
}

FPGA_STATUS _FPGA_GetHealthProperty_Configuration(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD cfgcrc_info;
    DWORD build_info;
    DWORD changeset, verbump, day, month, year, clean, tfsbuild;
    DWORD crc_errors;
    BOOL is_golden;
    DWORD role_id, role_version;
    DWORD shell_id, shell_version;
    DWORD sshell_id, sshell_version;
    DWORD board_id;

    char cfg_buf[1024];

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 104, &cfgcrc_info)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 59, &build_info)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 60, &changeset)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_IsGoldenImage(fpgaHandle, &is_golden, &role_id, &role_version)))
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 57, &board_id)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 64, &shell_id)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 58, &shell_version)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 99, &sshell_id)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 61, &sshell_version)) != FPGA_STATUS_SUCCESS)
        goto finish;


    crc_errors = cfgcrc_info & 0xFFFF;

    verbump = build_info & 0x7;
    day = (build_info >> 3) & 0x1F;
    month = (build_info >> 8) & 0xF;
    year = ((build_info >> 12) & 0xF) + 2013;
    clean = (build_info >> 30) & 0x1;
    tfsbuild = (build_info >> 31) & 0x1;

    sprintf_s(fpgaHealthProperty->name, "FPGA-CONFIG");
    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_CONFIGURATION;

    if (crc_errors == 0xFFFF && !(role_id == 0x601d && role_version == 0xCA7A0106))
    {
        //^^^ We ignore crc errors for early version of golden image that had SEU scrubbing disabled.
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_RMA;
        sprintf_s(fpgaHealthProperty->message, "ERROR: FPGA configuration SRAM ECC counter saturated.");
    }
    else if ((tfsbuild == 0) || (clean == 0))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: bitstream not built cleanly by TFS");
    }
    else
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK");
    }

    sprintf_s(cfg_buf, "[golden:%u,role_id:0x%x,role_ver:0x%x,shell_id:0x%x,shell_ver:0x%x,sshell_id:0x%x,sshell_ver:0x%x,crcerr:%u,chngset:%u,verbmp:%u,%u-%u-%u,clean:%u,tfs:%u]",
                            is_golden, role_id, role_version, shell_id, shell_version, sshell_id, sshell_version, crc_errors, changeset, verbump, year, month, day, clean, tfsbuild);
    strcat_s(fpgaHealthProperty->message, " ");
    strcat_s(fpgaHealthProperty->message, cfg_buf);

finish:
    return status;
}

FPGA_STATUS _FPGA_GetHealthProperty_DRAM(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD capabilities, ddr_present;
    DWORD ddr_status, ddr_ecc_counts;
    DWORD calib_success, calib_failed, calib_busy, ddr_reset;
    DWORD sbe, dbe;
    DWORD cycles_hi;
    DWORD shell_status;
    DWORD pll_locked;
    DWORD ddr_healthy;
    char dram_state_buf[1024];
    BOOL skip_state = FALSE;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 72, &capabilities)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 74, &ddr_ecc_counts)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 73, &ddr_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 67, &cycles_hi)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 68, &shell_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    ddr_present = (capabilities >> 1) & 0x1U;
    calib_success = ddr_status & 0x1;
    calib_failed = (ddr_status >> 1) & 0x1;
    calib_busy = (ddr_status >> 2) & 0x1;
    ddr_reset = (ddr_status >> 29) & 0x1;
    sbe = (ddr_ecc_counts >> 16) & 0xFFFFu;
    dbe = (ddr_ecc_counts >> 24) & 0xFFu;
    pll_locked = (shell_status >> 3) & 0x1;
    ddr_healthy = shell_status & 0x1;

    sprintf_s(fpgaHealthProperty->name, "FPGA-DRAM");
    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_DRAM;

    if (cycles_hi == 0)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK: initializing");
    }
    else if (ddr_present == 0)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK [DRAM disabled]");
        skip_state = TRUE;
    }
    else
    {
        if (calib_failed)
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_RMA;
            sprintf_s(fpgaHealthProperty->message, "ERROR: DRAM Calibration Failed");
        }
        else if (calib_busy || !calib_success || !pll_locked || !ddr_healthy)
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_UNKNOWN;
            sprintf_s(fpgaHealthProperty->message, "ERROR: DRAM Init Failed");
        }
        else if (sbe > 0)
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
            sprintf_s(fpgaHealthProperty->message, "WARNING: Correctable Single Bit Errors Detected");
        }
        else if (dbe > 0)
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_UNKNOWN;
            sprintf_s(fpgaHealthProperty->message, "ERROR: Uncorrectable Double Bit Errors Detected");
        }
        else
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
            sprintf_s(fpgaHealthProperty->message, "OK");
        }
    }

    if (!skip_state)
    {
        sprintf_s(dram_state_buf, "[present:%u,pll_lock:%u,ddr_healthy:%u,calib_success:%u,calib_failed:%u,calib_busy:%u,ddr_rst:%u,sbe:%u,dbe:%u]", ddr_present, pll_locked, ddr_healthy, calib_success, calib_failed, calib_busy, ddr_reset, sbe, dbe);
        strcat_s(fpgaHealthProperty->message, " ");
        strcat_s(fpgaHealthProperty->message, dram_state_buf);
    }

finish:
    return status;
}

FPGA_STATUS _FPGA_GetHealthProperty_ClockReset(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    char clk_buf[1024];
    DWORD shell_status, pll_locked;
    DWORD ctrl_status, force_app_reset, force_core_reset;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 68, &shell_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 0, &ctrl_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    force_app_reset = (ctrl_status >> 30) & 0x1;
    force_core_reset = (ctrl_status >> 31) & 0x1;
    pll_locked = (shell_status >> 2) & 0x1;

    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_CLOCKRESET;
    sprintf_s(fpgaHealthProperty->name, "FPGA-CLOCKRESET");

    if (!pll_locked)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_UNKNOWN;
        sprintf_s(fpgaHealthProperty->message, "ERROR: shell PLL did not lock");
    }
    else if ((force_app_reset || force_core_reset))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: software reset enabled");
    }
    else
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK");
    }

    sprintf_s(clk_buf, " [pll_locked:%u,force_app_reset:%u,force_core_reset:%u]", pll_locked, force_app_reset, force_core_reset);
    strcat_s(fpgaHealthProperty->message, clk_buf);

finish:
    return status;
}


FPGA_STATUS _FPGA_GetHealthProperty_Pcie(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    char pcie_buf[1024];
    char pcie_buf_2[1024];
    DWORD hip_id_reg, hip_links;
    DWORD hip_versions[2];
    DWORD active_lanes[2];
    DWORD link_speeds[2];
    DWORD hip_ids[2];
    DWORD dma_status;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 75, &hip_id_reg)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 76, &hip_versions[0])) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 77, &hip_versions[1])) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 69, &hip_links)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 34, &dma_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    active_lanes[0] = (hip_links) & 0xF;
    active_lanes[1] = (hip_links >> 8) & 0xF;
    link_speeds[0] = (hip_links >> 4) & 0xF;
    link_speeds[1] = (hip_links >> 12) & 0xF;
    hip_ids[0] = hip_id_reg & 0xFFFF;
    hip_ids[1] = (hip_id_reg >> 16) & 0xFFFF;

    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_PCIE;
    sprintf_s(fpgaHealthProperty->name, "FPGA-PCIE");

    //
    // Primary HIP must report > 0 lanes up.
    //
    if (active_lanes[0] == 0)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT;
        sprintf_s(fpgaHealthProperty->message, "ERROR: Primary HIP reporting no lanes up");
    }
    //
    // If primary HIP doesn't report minimum of 2x8, surface as warning. 
    //
    else if ((active_lanes[0] != 8) || (link_speeds[0] < 2))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: Primary HIP trained to less than 2x8");
    }
    //
    // If primary HIP is stream-based DMA, surface a warning if links didn't train to 3x8.
    //
    else if (((hip_ids[0] & 0xFF00) == 0x5E00) && ((active_lanes[0] != 8) || (link_speeds[0] != 3)))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: Primary HIP (stream DMA) did not train up to 3x8");
    }
    // Surface any DMA engine related errors as warnings
    else if ((dma_status >> 31) && ((dma_status & 0xF0) != 0))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: DMA engine reported problems");
    }
    //
    // If secondary HIP is stream-based DMA, surface warnings or errors based on link state.
    //
    else if ((hip_ids[1] & 0xFF00) == 0x5E00)
    {
        if ((active_lanes[1] == 0) || (link_speeds[1] == 0))
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT;
            sprintf_s(fpgaHealthProperty->message, "ERROR: Secondary HIP (stream DMA) reporting no lanes up");
        }
        else if ((active_lanes[1] != 8) || (link_speeds[1] != 3))
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
            sprintf_s(fpgaHealthProperty->message, "WARNING: Secondary HIP (stream DMA) did not train up to 3x8");
        }
        else
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
            sprintf_s(fpgaHealthProperty->message, "OK");
        }
    }
    //
    // If secondary HIP is slot-based DMA, surface warnings or errors based on link state.
    //
    else if (hip_ids[1] == 0x5107)
    {
        if ((active_lanes[1] == 0) || (link_speeds[1] == 0))
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT;
            sprintf_s(fpgaHealthProperty->message, "ERROR: Secondary HIP (stream DMA) reporting no lanes up");
        }
        else if ((active_lanes[1] != 8) || (link_speeds[1] != 2))
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
            sprintf_s(fpgaHealthProperty->message, "WARNING: Secondary HIP (stream DMA) did not train up to 2x8");
        }
        else
        {
            fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
            sprintf_s(fpgaHealthProperty->message, "OK");
        }
    }
    //
    // Healthy
    //
    else
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK");
    }

    sprintf_s(pcie_buf,   "[HIP-0,%ux%u,id:0x%x,ver:0x%x,dma_status:0x%x]", link_speeds[0], active_lanes[0], hip_ids[0], hip_versions[0], dma_status);
    strcat_s(fpgaHealthProperty->message, " ");
    strcat_s(fpgaHealthProperty->message, pcie_buf);

    if (hip_ids[1] != 0)
    {
        sprintf_s(pcie_buf_2, "[HIP-1,%ux%u,id:0x%x,ver:0x%x]", link_speeds[1], active_lanes[1], hip_ids[1], hip_versions[1]);
        strcat_s(fpgaHealthProperty->message, " ");
        strcat_s(fpgaHealthProperty->message, pcie_buf_2);
    }

finish:
    return status;
}
 
FPGA_STATUS _FPGA_GetHealthProperty_Network(_In_ FPGA_HANDLE fpgaHandle, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;
    DWORD cycles_hi;
    char net_buf[1024];

    DWORD board_id, no_network;
    DWORD tor_fcs_error_count, nic_fcs_error_count;
    DWORD tor_linkdown_count, nic_linkdown_count;
    DWORD tor_lane_state, nic_lane_state;

    DWORD tor_lanes_deskewed, nic_lanes_deskewed;
    DWORD tor_lanes_stable, nic_lanes_stable;
    DWORD tor_mac_hw_err, nic_mac_hw_err;
    DWORD tor_pcs_hw_err, nic_pcs_hw_err;

    DWORD tor_tx_count, tor_rx_count;
    DWORD nic_tx_count, nic_rx_count;

    DWORD soft_network_status;
    BOOL skip_state = FALSE;

    if (fpgaHandle == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle");
        goto finish;
    }

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 57, &board_id)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 67, &cycles_hi)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 21, &tor_fcs_error_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 22, &tor_linkdown_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 25, &nic_fcs_error_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 26, &nic_linkdown_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 42, &nic_lane_state)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 43, &tor_lane_state)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 19, &tor_tx_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 20, &tor_rx_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 23, &nic_tx_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 24, &nic_rx_count)) != FPGA_STATUS_SUCCESS)
        goto finish;

    if ((status = FPGA_ReadShellRegister(fpgaHandle, 4, &soft_network_status)) != FPGA_STATUS_SUCCESS)
        goto finish;

    no_network = (board_id == 0xB0) || (board_id == 0x0);
    tor_lanes_deskewed = tor_lane_state & 0x1;
    tor_lanes_stable = (tor_lane_state >> 1) & 0x1;
    tor_mac_hw_err = (tor_lane_state >> 2) & 0x7F;
    tor_pcs_hw_err = (tor_lane_state >> 9) & 0x1FF;

    nic_lanes_deskewed = nic_lane_state & 0x1;
    nic_lanes_stable = (nic_lane_state >> 1) & 0x1;
    nic_mac_hw_err = (nic_lane_state >> 2) & 0x7F;
    nic_pcs_hw_err = (nic_lane_state >> 9) & 0x1FF;

    fpgaHealthProperty->id = FPGA_HEALTH_PROPERTY_NETWORK;
    sprintf_s(fpgaHealthProperty->name, "FPGA-NETWORK");

    if (cycles_hi == 0)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK: initializing");
    }
    else if (no_network)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK [network disabled]");
        skip_state = TRUE;
    }
    else if ((tor_lanes_deskewed == 0) || (tor_lanes_stable == 0) || (nic_lanes_deskewed == 0) || (nic_lanes_stable == 0))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_ERROR_TRANSIENT;
        sprintf_s(fpgaHealthProperty->message, "ERROR: one or more network links down");
    }
    else if ((tor_linkdown_count > 100) || (nic_linkdown_count > 100))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: >100 link drops detected");
    }
    else if ((tor_fcs_error_count > 0) || (nic_fcs_error_count > 0))
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: corrupted packets detected");
    }
    else if (soft_network_status & ~0x1)
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_WARNING;
        sprintf_s(fpgaHealthProperty->message, "WARNING: soft shell network forcibly disabled");
    }
    else
    {
        fpgaHealthProperty->severity = FPGA_HEALTH_SEVERITY_OK;
        sprintf_s(fpgaHealthProperty->message, "OK");
    }

    if (!skip_state)
    {
        sprintf_s(net_buf, "[Soft-Network-Status:0x%x] [TOR-MAC,lanes_deskew:%u,lanes_stable:%u,mac_hw_err:0x%x,pcs_hw_err:0x%x,linkdrops:%u,rx_fcs_errs:%u,rx_count:%u,tx_count:%u] [NIC-MAC,lanes_deskew:%u,lanes_stable:%u,mac_hw_err:0x%x,pcs_hw_err:0x%x,linkdrops:%u,rx_fcs_errs:%u,rx_count:%u,tx_count:%u]",
            soft_network_status,
            tor_lanes_deskewed, tor_lanes_stable, tor_mac_hw_err, tor_pcs_hw_err, tor_linkdown_count, tor_fcs_error_count, tor_rx_count, tor_tx_count,
            nic_lanes_deskewed, nic_lanes_stable, nic_mac_hw_err, nic_pcs_hw_err, nic_linkdown_count, nic_fcs_error_count, nic_rx_count, nic_tx_count);

        strcat_s(fpgaHealthProperty->message, " ");
        strcat_s(fpgaHealthProperty->message, net_buf);
    }

finish:
    return status;
}

_CDLLEXPORT_ FPGA_STATUS FPGA_GetHealthProperty(_In_ FPGA_HANDLE fpgaHandle, _In_ DWORD healthPropertyId, _Out_ FPGA_HEALTH_PROPERTY *fpgaHealthProperty)
{
    FPGA_STATUS status = FPGA_STATUS_SUCCESS;

	if (fpgaHandle == NULL)
	{
		LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA handle.");
		goto finish;
	}

    if (fpgaHealthProperty == NULL)
    {
        LOG_ERROR(status, FPGA_STATUS_NULL_POINTER_GIVEN, "Null pointer given for FPGA health property");
        goto finish;
    }

    switch (healthPropertyId)
    {
        case FPGA_HEALTH_PROPERTY_CLOCKRESET: return _FPGA_GetHealthProperty_ClockReset(fpgaHandle, fpgaHealthProperty);
        case FPGA_HEALTH_PROPERTY_CONFIGURATION: return _FPGA_GetHealthProperty_Configuration(fpgaHandle, fpgaHealthProperty);
        case FPGA_HEALTH_PROPERTY_DRAM: return _FPGA_GetHealthProperty_DRAM(fpgaHandle, fpgaHealthProperty);
        case FPGA_HEALTH_PROPERTY_NETWORK: return _FPGA_GetHealthProperty_Network(fpgaHandle, fpgaHealthProperty);
        case FPGA_HEALTH_PROPERTY_PCIE: return _FPGA_GetHealthProperty_Pcie(fpgaHandle, fpgaHealthProperty);
        case FPGA_HEALTH_PROPERTY_TEMPERATURE: return _FPGA_GetHealthProperty_Temperature(fpgaHandle, fpgaHealthProperty);
        default:
            LOG_ERROR(status, FPGA_STATUS_BAD_ARGUMENT, "Invalid argument healthPropertyId");
            goto finish;
    }

finish:
    return status;
}
