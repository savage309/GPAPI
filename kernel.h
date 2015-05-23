#pragma once

#include "common.h"

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {
    struct Kernel {
    private:
        GPU_KERNEL kernel;
    public:
        Kernel() {kernel=NULL;};
        GPU_KERNEL get() {
            return kernel;
        }
        void freeMem() {
            GPU_RESULT err = GPU_SUCCESS;
            if (kernel) {
                
#ifdef TARGET_OPENCL
                clReleaseKernel(kernel);
#endif
#ifdef TARGET_CUDA
#endif
                kernel = NULL;
            }
            CHECK_ERROR(err);

        }
        ~Kernel() {
            freeMem();
        }
        void init(const char* name, GPU_PROGRAM program) {
            freeMem();
            
            if (kernel)
                return;
            
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            kernel = clCreateKernel(program, name, &err);
            CHECK_ERROR(err);
#endif //TARGET_OPENCL
#ifdef TARGET_CUDA
            err = cuModuleGetFunction(&kernel, program, name);
            CHECK_ERROR(err);
#endif
            CHECK_ERROR(err);
        }
    };

}