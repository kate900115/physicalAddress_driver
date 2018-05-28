
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/ioport.h>
#include <linux/list.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/interrupt.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
//#include <linux/of.h>
//#include <linux/of_platform.h>
#include <asm/io.h>

#include <asm/uaccess.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/interrupt.h>

#include "gpumemdrv.h"
#include "ioctlrw.h"
#include "gpumemioctl.h"
#include "gpumemproc.h"
#include <linux/memremap.h>
#include <linux/delay.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/errno.h>
#include <linux/ioctl.h>
#include <linux/cdev.h>



//-----------------------------------------------------------------------------

MODULE_AUTHOR("Vladimir Karakozov. karakozov@gmail.com");
MODULE_LICENSE("GPL");

//-----------------------------------------------------------------------------
static struct gpumem dev;

//-----------------------------------------------------------------------------

static struct gpumem *file_to_device( struct file *file )
{
    return (struct gpumem*)file->private_data;
}

//--------------------------------------------------------------------

static int gpumem_open( struct inode *inode, struct file *file )
{
    file->private_data = (void*)&dev;
    return 0;
}

//-----------------------------------------------------------------------------

static int gpumem_close( struct inode *inode, struct file *file )
{
    file->private_data = 0;
    return 0;
}

//-----------------------------------------------------------------------------

static long gpumem_ioctl( struct file *file, unsigned int cmd, unsigned long arg )
{
    int error = 0;
    if (cmd==IOCTL_V2P) {
	pr_info("[gpumem_ioctl] The ioctl_v2p_convert(arg) is called.\n");
	error = ioctl_v2p_convert(arg);
    }
    if (cmd==IOCTL_P2V){
	pr_info("[gpumem_ioctl] The ioctl_p2v_convert(arg) is called.\n");
	error = ioctl_p2v_convert(arg);
    }

    return error;
}

//-----------------------------------------------------------------------------

int gpumem_mmap(struct file *file, struct vm_area_struct *vma)
{

	pr_info("[mmap] I'm in mmap\n");

	size_t size = vma->vm_end - vma->vm_start;

	if (!(vma->vm_flags & VM_MAYSHARE)){
		return -EINVAL;
	}
	
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	
	//    void* kmalloc_area = kmalloc(512, GFP_USER);
	//    if (kmalloc_area==NULL){
	//	pr_info("@@@@@ kmalloc failed!");
	//    }
	//uint64_t phyAddr = virt_to_phys(kmalloc_area);

	
	// read value from savedPhysAddr
	
	struct savedAddress savedAddr;
	savedAddr = savedPhysAddr(0, 0, 1);

	pr_info("[mmap] read previously saved address from kernel\n");
	pr_info("[mmap] physical address = %ld\n", savedAddr.addr);

	if (savedAddr.op==1){
		//p2v
	//	void* ioremap_area = ioremap(savedAddr.addr, size);
	//	if (ioremap_area==NULL){
	//		pr_info("[mmap] ioremap unsuccessful\n");
	//		return -ENXIO;
	//	}
		
		//int errorCode = remap_pfn_range(vma, vma->vm_start+savedAddr.addr, savedAddr.addr, size, PAGE_SHARED);  
		int errorCode = remap_pfn_range(vma, vma->vm_start, savedAddr.addr>>PAGE_SHIFT, size, PAGE_SHARED);  


		// write noop to saved address operation element
		savedPhysAddr(savedAddr.addr, 0, 0);
	 
		if (errorCode){
			return -ENXIO;
		}

		return 0;
	
	}

	else if (savedAddr.op==0){
		//v2p
		void* kmalloc_area = kmalloc(size, GFP_USER);
		uint64_t phyaddr = virt_to_phys(kmalloc_area);
		savedPhysAddr(phyaddr,2,0);

		if (kmalloc_area==NULL){
			pr_info("[mmap] kmalloc area failed\n");
			return -ENXIO;
		}
		//int errorCode = remap_pfn_range(vma, vma->vm_start+phyaddr, phyaddr, size, PAGE_SHARED);
		int errorCode = remap_pfn_range(vma, vma->vm_start, phyaddr>>PAGE_SHIFT, size, PAGE_SHARED);  


		if (errorCode){
			return -ENXIO;
		}

		return 0;

	}


}

//-----------------------------------------------------------------------------

struct file_operations gpumem_fops = {

    .owner = THIS_MODULE,
    .unlocked_ioctl = gpumem_ioctl,
    .compat_ioctl = gpumem_ioctl,
    .open = gpumem_open,
    .release = gpumem_close,
    .mmap = gpumem_mmap,
};

//-----------------------------------------------------------------------------

static struct miscdevice gpumem_dev = {

    MISC_DYNAMIC_MINOR,
    GPUMEM_DRIVER_NAME,
    &gpumem_fops
};

//-----------------------------------------------------------------------------

static int __init gpumem_init(void)
{
    pr_info(GPUMEM_DRIVER_NAME ": %s()\n", __func__);
    dev.proc = 0;
    sema_init(&dev.sem, 1);
    INIT_LIST_HEAD(&dev.table_list);
    //zyuxuan
    gpumem_register_proc(GPUMEM_DRIVER_NAME, 0, &dev);
    //gpumem_register_proc("Catapult FPGA", 0, &dev);

    misc_register(&gpumem_dev);
    return 0;
}

//-----------------------------------------------------------------------------

static void __exit gpumem_cleanup(void)
{
    pr_info(GPUMEM_DRIVER_NAME ": %s()\n", __func__);
    gpumem_remove_proc(GPUMEM_DRIVER_NAME);
    misc_deregister(&gpumem_dev);
}

//-----------------------------------------------------------------------------

module_init(gpumem_init);
module_exit(gpumem_cleanup);

//-----------------------------------------------------------------------------
