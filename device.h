
#ifndef opencl_device_h
#define opencl_device_h
#include "common.h"

#ifndef __GPAPI_H__
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {
    
    struct Device {
        std::string name;
        
        Platform platform;
        DeviceID device;
        Context context;
        Program program;
        Kernel kernel;
        Queue queue;
        
        virtual void init(Platform platformId, DeviceID deviceId, std::string nameId, Context contextId, Program programId) {
            platform = platformId;
            device = deviceId;
            context = contextId;
            program = programId;
            name = nameId;
            
            queue.init(device, context);
        }
        virtual void launchKernel() = 0;
        virtual void freeMem() = 0;
        virtual ~Device(){}
    };
}

#endif
