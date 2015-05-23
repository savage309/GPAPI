#pragma once

#include "common.h"

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {

    /*! \brief Represents device memory buffer
     */
    struct Buffer {
public:
        
    /*! Creates empty device memory buffer (does not alloce/transfer anything) */
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
     
    /*! \brief Allocates numBytes device memory and if hostSrc is not NULL, transfer numBytes memory from hostSrc to the allocated device memory. Each device memory is allocated per device; queue and context are used to specify that device.
     \param queue Should be from the result of Device::getQueue() method
     \param context Should be from the result of Device::getContext() method
     \param hostSrc If != NULL, this method will allocate numBytes of device memory and will transfer numBytes hostSrc memory to the allocated device memory. If hostSrc is NULL, transfer is not made (only allocation)
     \param numBytes Number of bytes to allocate(and possibly transfer). Should be > 0
     */
    void init(GPU_QUEUE queue, GPU_CONTEXT context, const void* hostSrc, size_t numBytes) {
        if (numBytes == 0)
            return;
        
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
    
    /*! \return Returns pointer to the allocated *DEVICE* memory. Note that this pointer may not be valid host pointer (it might be valid only if TARGET_NATIVE is defined). If you need to read the device memory, use have to use Buffer::download method. */
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
    
    /*! \brief Frees the device memory, if there is any allocated such. May be called multiple times */
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
        
    /*! \brief Transfers allocated device memory to the host 
     \param queue Should be from the result of Device::getQueue() method
     \param context Should be from the result of Device::getContext() method
     \param hostPtr Pointer to host memory (should points to at least 'bytes' bytes)
     \param bytes Number of bytes that should be transfered from the device to the host. Shoud be > 0.
     */
    void download(GPU_QUEUE queue, Context context, void* hostPtr, size_t bytes) {
        if (bytes == 0)
            return;
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
private:
#ifdef TARGET_OPENCL
    ///pointer to OpenCL device memory
    cl_mem clMem;
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
    ///pointer to CUDA device memory
    CUdeviceptr cudaMem;
#endif //TARGET_CUDA
#ifdef TARGET_NATIVE
    ///pointer to NATIVE device memory
    void* nativeMem;
#endif //TARGET_NATIVE
};
}

