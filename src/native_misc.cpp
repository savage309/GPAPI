
#include "native_misc.h"

#ifdef TARGET_NATIVE

#define __NATIVE__
#include "queue.h"
#include "buffer.h"
#include "kernel.h"
#include "kernel_launch.h"


static int threadIdx;
#include "../kernel.cl"

enum KernelParams {
    D_A = 0,
    D_B,
    D_RES,
    COUNT,
};

void GPAPI::NativeDevice::launchKernel(KernelLaunch& kernelLaunch, size_t numTasks) {
    for (int i = 0; i < numTasks; ++i) {
        threadIdx = i;
        
        int** paramsPtrs = (int**)kernelLaunch.ptrs;
        
        vecAdd(paramsPtrs[D_A],
               paramsPtrs[D_B],
               paramsPtrs[D_RES],
               *(int*)paramsPtrs[COUNT]);
    }
}

GPAPI::NativeDevice* GPAPI::getNativeDevice(){
    static NativeDevice nativeDevice;
    return &nativeDevice;
}


#endif