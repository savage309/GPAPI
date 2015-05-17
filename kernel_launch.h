#ifndef opencl_kernel_launch_h
#define opencl_kernel_launch_h

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

#include "common.h"
#include "native_misc.h"

namespace GPAPI {
    struct KernelLaunch {
#if (defined TARGET_CUDA)
        char paramsBuffer[4096]; // A buffer to hold parameter values
        void *paramsPtrs[1024]; // A buffer to hold pointers to each parameter
        int numParams;
        int paramOffset;
#endif
#ifdef TARGET_NATIVE
        int numParams;
        std::vector<void*> ptrsToDelete;
        void* ptrs[1024];
#endif
        
        Kernel& kernel;
        int index;
        KernelLaunch(Kernel& kernel):kernel(kernel), index(0) {
#if  (defined TARGET_CUDA)
            numParams = paramOffset = 0;
#endif
#if (defined TARGET_NATIVE)
            numParams = 0;
#endif
        
        }
        void addArg(Buffer& buffer) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(cl_mem), buffer.get());
#endif
#if  (defined TARGET_CUDA)
            size_t argSize = sizeof(buffer.get());
            
            printf("%p\n", buffer.get());
            
            paramsPtrs[numParams++]=(paramsBuffer+paramOffset);
            memcpy(paramsBuffer+paramOffset, buffer.get(), argSize);
            paramOffset+=argSize;
#endif
#ifdef TARGET_NATIVE
            ptrs[numParams] = buffer.get();
            numParams++;
#endif
            CHECK_ERROR(err);
        }
        void addArg(int arg) {
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            err = clSetKernelArg(kernel.get(), index++, sizeof(unsigned int), &arg);
#endif
#if  (defined TARGET_CUDA)
            size_t argSize = sizeof(arg);
            paramsPtrs[numParams++]=(paramsBuffer+paramOffset);
            memcpy(paramsBuffer+paramOffset, &arg, argSize);
            paramOffset+=argSize;
#endif
#ifdef TARGET_NATIVE
            int* newInt = new int;
            *newInt = arg;
            ptrs[numParams] = newInt;
            numParams++;
            ptrsToDelete.push_back(newInt);
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
                               (unsigned int)globalSize, 1UL, 1UL, // grid size
                               (unsigned int)localSize, 1UL, 1UL, // block size
                               0, // shared size
                               NULL, // stream
                               &paramsPtrs[0],
                               NULL
                               );
            popContext(context);
#endif
#ifdef TARGET_NATIVE
            NativeDevice* device = getNativeDevice();
            device->launchKernel(*this, globalSize * localSize);
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
        void freeMem() {
#ifdef TARGET_NATIVE
            for (int i = 0; i < ptrsToDelete.size(); ++i)
                delete (char*)ptrsToDelete[i];
#endif
        }
        ~KernelLaunch() {
            freeMem();
        }
    };

}

#endif
