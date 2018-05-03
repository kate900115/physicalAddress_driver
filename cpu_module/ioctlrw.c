
#include <linux/kernel.h>
#define __NO_VERSION__
#include <linux/module.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/pagemap.h>
#include <linux/interrupt.h>
#include <linux/proc_fs.h>
#include <asm/io.h>

#include "gpumemdrv.h"
#include "gpumemioctl.h"

//-----------------------------------------------------------------------------

int get_nv_page_size(int val)
{
    switch(val) {
    case NVIDIA_P2P_PAGE_SIZE_4KB: return 4*1024;
    case NVIDIA_P2P_PAGE_SIZE_64KB: return 64*1024;
    case NVIDIA_P2P_PAGE_SIZE_128KB: return 128*1024;
    }
    return 0;
}

//--------------------------------------------------------------------

void free_nvp_callback(void *data)
{
    int res;
    struct gpumem_t *entry = (struct gpumem_t*)data;
    if(entry) {
        res = nvidia_p2p_free_page_table(entry->page_table);
        if(res == 0) {
            printk(KERN_ERR"%s(): nvidia_p2p_free_page_table() - OK!\n", __FUNCTION__);
            //entry->virt_start = 0ULL;
            //entry->page_table = 0;
        } else {
            printk(KERN_ERR"%s(): Error in nvidia_p2p_free_page_table()\n", __FUNCTION__);
        }
    }
}


//zyuxuan
int ioctl_mem_convert(unsigned long arg){
	int error = 0;
	// to copy the argument from user space to kernel space
	struct cpuaddr_state_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_state_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}

	void* address = addr.handle;
	phys_addr_t paddr = virt_to_phys(address);
	addr.paddr = paddr;

 	pr_info("@@@@I'm ioctl_mem_convert\n");	
	pr_info("physical address = %ld\n", addr.paddr);


	if (copy_to_user((void*)arg, &addr, sizeof(struct cpuaddr_state_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n",__FUNCTION__);
		error = -EFAULT;
		return error;
	}

	return error;
}

