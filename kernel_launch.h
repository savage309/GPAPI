//
//  kernel_launch.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_kernel_launch_h
#define opencl_kernel_launch_h

#include "common.h"

namespace GPAPI {
    struct KernelLaunch {
        Kernel& kernel;
        int index;
        KernelLaunch(Kernel& kernel):kernel(kernel), index(0) {}
        void addArg(Buffer& buffer) {
            GPU_RESULT err;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(cl_mem), buffer.get());
            CHECK_ERROR(err);
#endif
        }
        void addArg(int arg) {
            GPU_RESULT err;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(unsigned int), &arg);
            CHECK_ERROR(err);
#endif
        }
        
        void run(Queue queue, size_t& globalSize, size_t& localSize) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, NULL, &globalSize, &localSize,
                                         0, NULL, NULL);
#endif
            CHECK_ERROR(err);
        }
        void wait() {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TRAGET_OPENCL
            err = clFinish(queue.get());
#endif
            CHECK_ERROR(err);
        }
    };

}

#endif
