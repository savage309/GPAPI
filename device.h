//
//  device.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_device_h
#define opencl_device_h
#include "common.h"
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
        virtual ~Device(){}
    };
}

#endif
