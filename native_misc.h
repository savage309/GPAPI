//
//  native_device.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef __GPAPI_NATIVE_MISC_H__
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

#endif
