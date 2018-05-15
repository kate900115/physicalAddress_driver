
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
    if (cmd==IOCTL_GPUMEM_LOCK) {
	error = ioctl_mem_convert(arg);
	pr_info("@@@@@@@@@@ The ioctl_mem_convert(arg) is called.\n");
    }
    if (cmd==IOCTL_GPUMEM_UNLOCK){
	error = ioctl_p2v_convert(arg);
	pr_info("@@@@@@@@@@ The ioctl_p2v_convert(arg) is called.\n");
    }

    return error;
}

//-----------------------------------------------------------------------------

int gpumem_mmap(struct file *file, struct vm_area_struct *vma)
{
	size_t size = vma->vm_end - vma->vm_start;

	if (!(vma->vm_flags & VM_MAYSHARE)){
		return -EINVAL;
	}
	
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	
	//    void* kmalloc_area = kmalloc(512, GFP_USER);

	//    if (kmalloc_area==NULL){
	//	pr_info("@@@@@ kmalloc failed!");
	//    }
	//vma->vm_flags = vma->vm_flags | VM_LOCKED；
	//uint64_t phyAddr = virt_to_phys(kmalloc_area);

	pr_info("I'm in mmap\n");

	// read value from savedPhysAddr
	uint64_t a = savedPhysAddr(0, 1);
	pr_info("physical address = %ld\n", a);

	// write to savedPhysAddr to be 0
	savedPhysAddr(0,0);

	if (a!=0){
		void* ioremap_area = ioremap(a, 4096);

		if (ioremap_area==NULL){
			pr_info("ioremap unsuccessful\n");
		}

		pr_info("physical address = %ld\n", a);
		int errorCode = remap_pfn_range(vma, vma->vm_start, a, size, PAGE_SHARED);  
	 
		if (errorCode){
			return -ENXIO;
		}

		return 0;
	}
	else{
		void* kmalloc_area = kmalloc(size, GFP_USER);
		uint64_t phyaddr = virt_to_phys(kmalloc_area);
		int errorCode = remap_pfn_range(vma, vma->vm_start, phyaddr, size, PAGE_SHARED);
		if (errorCode){
			return -ENXIO;
		}

		savedPhysAddr(phyaddr,0);
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
