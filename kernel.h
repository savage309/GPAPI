

#ifndef opencl_kernel_h
#define opencl_kernel_h

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
        void init(const char* name, GPU_PROGRAM program) {
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

#endif
