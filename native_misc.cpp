//
//  native_misc.cpp
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#include "native_misc.h"

#ifdef TARGET_NATIVE

#define __NATIVE__
static int threadIdx = 0;
#include "queue.h"
#include "buffer.h"
#include "kernel.h"
#include "kernel_launch.h"
#include "kernel.cl"


enum KernelParams {
    D_A = 0,
    D_B,
    D_RES,
    COUNT,
};

void GPAPI::NativeDevice::launchKernel(KernelLaunch& kernelLaunch, size_t numTasks){
    for (int i = 0; i < numTasks; ++i) {
        threadIdx = i;
        vecAdd((int*)kernelLaunch.paramsPtrs[KernelParams::D_A],
               (int*)kernelLaunch.paramsPtrs[KernelParams::D_B],
               (int*)kernelLaunch.paramsPtrs[KernelParams::D_RES],
               *(int*)kernelLaunch.paramsPtrs[KernelParams::COUNT]);
    }
}

GPAPI::NativeDevice* GPAPI::getNativeDevice(){
    static NativeDevice nativeDevice;
    return &nativeDevice;
}


#endif