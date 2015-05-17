//
//  queue.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_queue_h
#define opencl_queue_h
#include "common.h"

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
            clReleaseCommandQueue(queue);
            queue = NULL;
#endif
        }
        
        GPU_QUEUE get() {
            return queue;
        }
    };
}

#endif
