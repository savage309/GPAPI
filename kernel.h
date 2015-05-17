//
//  kernel.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_kernel_h
#define opencl_kernel_h

#include "common.h"

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
        }
    };

}

#endif
