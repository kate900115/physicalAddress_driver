
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

#include "v2p2vdrv.h"
#include "v2p2vioctl.h"

//-----------------------------------------------------------------------------


struct savedAddress savedPhysAddr(uint64_t addr, int op, bool isRead ){

	static int times = 0;
	times++;
	pr_info("[savedPhysAddr] the function is called %d times\n",times);

	static struct savedAddress SavedAddr;

	if (isRead){
		return SavedAddr;
	}
	else{
		SavedAddr.addr = addr;
		SavedAddr.op = op;
		return SavedAddr;
	}

}


//zyuxuan
int ioctl_v2p_convert(unsigned long arg){
	pr_info("[ioctl_v2p] I'm ioctl_v2p_convert\n");	
	int error = 0;
	// to copy the argument from user space to kernel space
	struct cpuaddr_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}
	
	// read the address from saved address
	struct savedAddress temp = savedPhysAddr(0,0,1);
	addr.paddr = temp.addr;

 	pr_info("[ioctl_v2p] previously saved physical address = %ld\n", addr.paddr);
	
	savedPhysAddr(0,0,0);

	pr_info("[ioctl_v2p] write 0 to saved address\n");

	if (copy_to_user((void*)arg, &addr, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n",__FUNCTION__);
		error = -EFAULT;
		return error;
	}

	return error;
}

//zyuxuan
int ioctl_p2v_convert(unsigned long arg){
	pr_info("[ioctl_p2v] I'm ioctl_p2v_convert\n");	
	int error = 0;
	// to copy the argument from user space to kernel space
	struct cpuaddr_t addr;
	if (copy_from_user(&addr, (void*)arg, sizeof(struct cpuaddr_t))){
		printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
		error = -EFAULT;
		return error;
	}

	savedPhysAddr(addr.paddr, 1, 0);
 	pr_info("[ioctl_p2v] now is saving physical address = %d\n", addr.paddr);

	return error;
}


