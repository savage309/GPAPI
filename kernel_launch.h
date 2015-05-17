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
/*
 void addArg(void *arg, int argSize) { paramsPtrs[numParams++]=(paramsBuffer+paramOffset); memcpy(paramsBuffer+paramOffset, arg, argSize); paramOffset+=argSize; }
	// New method of setting arguments
	GPUResult addParam(OpenCLBuffer &buffer) {
 //
 if (isCudaCpu) {
 cudaCPUParams.paramPointers[cudaCPUParams.paramsCount]=buffer.cudaCPUbuf;
 cudaCPUParams.paramsCount++;
 }
 //
 
 addArg(&buffer.cu_buf, sizeof(buffer.cu_buf));
 return GPU_SUCCESS;
	}
 */
namespace GPAPI {
    struct KernelLaunch {
#ifdef TARGET_CUDA
        char paramsBuffer[4096]; // A buffer to hold parameter values
        void *paramsPtrs[1024]; // A buffer to hold pointers to each parameter
        int numParams;
        int paramOffset;
#endif
        Kernel& kernel;
        int index;
        KernelLaunch(Kernel& kernel):kernel(kernel), index(0) {
#ifdef TARGET_CUDA
            numParams = paramOffset = 0;
#endif
        
        }
        void addArg(Buffer& buffer) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(cl_mem), buffer.get());
#endif
#ifdef TARGET_CUDA
            size_t argSize = sizeof(buffer.get());
            paramsPtrs[numParams++]=(paramsBuffer+paramOffset);
            memcpy(paramsBuffer+paramOffset, buffer.get(), argSize);
            paramOffset+=argSize;
#endif
            CHECK_ERROR(err);
        }
        void addArg(int arg) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(unsigned int), &arg);
#endif
#ifdef TARGET_CUDA
            size_t argSize = sizeof(arg);
            paramsPtrs[numParams++]=(paramsBuffer+paramOffset);
            memcpy(paramsBuffer+paramOffset, &arg, argSize);
            paramOffset+=argSize;
#endif
            CHECK_ERROR(err);

        }
        
        void run(Queue queue, Context context, size_t& globalSize, size_t& localSize) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, NULL, &globalSize, &localSize,
                                         0, NULL, NULL);
#endif
#ifdef TARGET_CUDA
            pushContext(context);
            err = cuLaunchKernel(kernel.get(),
                               (unsigned long)globalSize, 1UL, 1UL, // grid size
                               (unsigned long)localSize, 1UL, 1UL, // block size
                               0, // shared size
                               NULL, // stream
                               &paramsPtrs[0],
                               NULL
                               );
            popContext(context);
#endif
            CHECK_ERROR(err);
        }
        void wait(Queue queue, Context context) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clFinish(queue.get());
#endif
#ifdef TARGET_CUDA
            pushContext(context);
            err = cuCtxSynchronize();
            CHECK_ERROR(err);
            popContext(context);
#endif
            CHECK_ERROR(err);
        }
    };

}

#endif
