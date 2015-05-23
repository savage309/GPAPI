#pragma once

#include "common.h"

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {
    struct Queue {
    private:
        GPU_QUEUE queue;
    public:
        Queue() { queue = NULL; };
        
        void init(GPU_DEVICE device, GPU_CONTEXT context) {
            freeMem();
            
            GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
            cl_command_queue_properties queueFlags;
            err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queueFlags), &queueFlags, NULL);
            CHECK_ERROR(err);
            
            queueFlags &= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            
            queue = clCreateCommandQueue(context, device, 0, &err);
#endif
            CHECK_ERROR(err);

        }
        void freeMem() {
#ifdef TARGET_OPENCL
           
#endif
        }
        
        GPU_QUEUE get() {
            return queue;
        }
        
        ~Queue() {
#ifdef TARGET_OPENCL
             clReleaseCommandQueue(queue);
             queue = NULL;
#endif
        }
        
    private:
        Queue(const Queue&);
        Queue& operator=(const Queue&);
    };
}