
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


uint64_t savedPhysAddr(uint64_t addr, bool isRead ){
	static uint64_t AddrSaved = 0;
	static int a = 0;
	a++;
	pr_info("the function is called a=%d times\n",a);
	pr_info("saved addr = %ld\n", AddrSaved);
	if (isRead) return AddrSaved;
	AddrSaved = addr;
	return 0;

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
	addr.paddr = savedPhysAddr(0,1);
	savedPhysAddr(0,0);

 	pr_info("@@@@I'm ioctl_mem_convert\n");	
	pr_info("physical address = %ld\n", addr.paddr);


	if (copy_to_user((void*)arg, &addr, sizeof(struct cpuaddr_state_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n",__FUNCTION__);
		error = -EFAULT;
		return error;
	}

	return error;
}

//zyuxuan
int ioctl_p2v_convert(unsigned long arg){
	int error = 0;
	// to copy the argument from user space to kernel space
	struct cpuaddr_state_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_state_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}


	uint64_t a = savedPhysAddr(addr.paddr, 0);
 	pr_info("@@@@I'm ioctl_p2v_convert\n");	
	pr_info("physical address = %ld\n", addr.paddr);
	
	



	return error;
}


