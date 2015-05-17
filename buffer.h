
#ifndef opencl_buffer_h
#define opencl_buffer_h

#include "common.h"

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {
struct Buffer {
#ifdef TARGET_OPENCL
    cl_mem clMem;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
    CUdeviceptr cudaMem;
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
        GPU_RESULT err = GPU_SUCCESS;

#ifdef TARGET_OPENCL
        clMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numBytes, NULL, &err);
        CHECK_ERROR(err);
        if (hostSrc) {
            err = clEnqueueWriteBuffer(queue, clMem, GPU_TRUE, 0, numBytes, hostSrc, 0, NULL, NULL);
            CHECK_ERROR(err);
        }
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        pushContext(context);
        err = cuMemAlloc(&cudaMem, numBytes);
        CHECK_ERROR(err);
        if (hostSrc) {
            err = cuMemcpyHtoD(cudaMem, hostSrc, numBytes);
            CHECK_ERROR(err);
        }
        popContext(context);
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        nativeMem = new char[numBytes];
        if (hostSrc) {
            memcpy(nativeMem, hostSrc, numBytes);
        }
#endif //TARGET_NATIVE
        CHECK_ERROR(err);
    }
    
    void* get() {
#ifdef TARGET_OPENCL
        return &clMem;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
        return &cudaMem;
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
        if (cudaMem)
            err = cuMemFree(cudaMem);
        CHECK_ERROR(err);
        cudaMem = NULL;
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        delete[] (char*)nativeMem;
        nativeMem = NULL;
#endif //TARGET_NATIVE
        CHECK_ERROR(err);
    }
    void download(GPU_QUEUE queue, Context context, void* hostPtr, size_t bytes) {
        GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
        err = clEnqueueReadBuffer(queue, *(cl_mem*)get(), GPU_TRUE, 0, bytes, hostPtr, 0, NULL, NULL );
#endif
#ifdef TARGET_CUDA
        pushContext(context);
        err = cuMemcpyDtoH(hostPtr, cudaMem, bytes);
        popContext(context);
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
        memcpy(hostPtr, nativeMem, bytes);
#endif //TARGET_NATIVE
        CHECK_ERROR(err);

    }
    ~Buffer() {
        freeMem();
    }
};
}


#endif
