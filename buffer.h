//
//  buffer.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_buffer_h
#define opencl_buffer_h

#include "common.h"

#if !(defined TARGET_OPENCL) && !(defined TARGET_CUDA) && !(defined TARGET_NATIVE)
#error GPAPI::Buffer needs one of CUDA, OpenCL or NATIVE targets to be defined
#endif

namespace GPAPI {
struct Buffer {
#ifdef TARGET_OPENCL
    cl_mem clMem;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
    CUdeviceptr* cudaMem;
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
    void* nativeMem;
#endif //TARGET_NATIVE
    
    Buffer() {
#ifdef TARGET_OPENCL
        clMem = NULL;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        cudaMem = NULL;
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        nativeMem = NULL;
#endif //TARGET_NATIVE
        
    }
    
    void init(GPU_QUEUE queue, GPU_CONTEXT context, const void* hostSrc, size_t numBytes) {
        freeMem();
        
#ifdef TARGET_OPENCL
        GPU_RESULT err = GPU_SUCCESS;
        clMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numBytes, NULL, &err);
        CHECK_ERROR(err);
        if (hostSrc) {
            err = clEnqueueWriteBuffer(queue, clMem, GPU_TRUE, 0,
                                       numBytes, hostSrc, 0, NULL, NULL);
            CHECK_ERROR(err);
        }
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        nativeMem = new char[numBytes];
        if (hostSrc) {
            memcpy(nativeMem, hostSrc, numBytes);
        }
#endif //TARGET_NATIVE
    }
    
    void* get() {
#ifdef TARGET_OPENCL
        return &clMem;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        return cudaMem;
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        return nativeMem;
#endif //TARGET_NATIVE
    }
    
    void freeMem() {
        GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
        if (clMem)
            err = clReleaseMemObject(clMem);
        clMem = NULL;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        delete[] nativeMem;
        nativeMem = NULL;
#endif //TARGET_NATIVE
        CHECK_ERROR(err);
    }
    ~Buffer() {
        freeMem();
    }
};
}


#endif
