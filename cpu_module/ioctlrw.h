
#ifndef _IOCTLRW_H_
#define _IOCTLRW_H_

//-----------------------------------------------------------------------------

int ioctl_mem_lock(struct gpumem *drv, unsigned long arg);
int ioctl_mem_unlock(struct gpumem *drv, unsigned long arg);
int ioctl_mem_state(struct gpumem *drv, unsigned long arg);

//zyuxuan
int ioctl_mem_convert(unsigned long arg);
int ioctl_p2v_convert(unsigned long arg);
//-----------------------------------------------------------------------------

#endif //_IOCTLRW_H_
