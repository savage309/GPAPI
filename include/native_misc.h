#pragma once
#define __GPAPI_NATIVE_MISC_H__

#include "configure.h"
#include "common.h"

namespace GPAPI {
#ifdef TARGET_NATIVE

    struct KernelLaunch;
    
    struct NativeDevice {
        void launchKernel(KernelLaunch& kernelLaunch, size_t numTasks);
    };
    
    NativeDevice* getNativeDevice();
    
#endif
}
