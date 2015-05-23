#pragma once

#include "common.h"

#ifndef __GPAPI_H__
#   error For GPAPI you need only to include gpapi.h
#endif

#include "init_params.h"

#include <cassert>

namespace GPAPI {
    
    struct Device {
        std::string name;
        typedef  InitParams::VendorParams::VendorType VendorType;
        typedef InitParams::VendorParams::DeviceType DeviceType;
        
        virtual void init(Platform platformId, DeviceID deviceId, std::string nameId, Context contextId, Program programId, VendorType vendorTypeId, DeviceType deviceTypeId, size_t localMemSizeId, size_t threadPerBlockId) {
            freeMem();
            maxLocalMemSize = localMemSizeId;
            maxThreadsPerBlock = threadPerBlockId;
            
            platform = platformId;
            device = deviceId;
            context = contextId;
            program = programId;
            name = nameId;
            
            queue.init(device, context);
        }
        
        void launchKernel(size_t globalSize, size_t localSize) {
            kernelLaunch.run(queue.get(), context, globalSize, localSize);
        };
        void freeMem() {
            for (int i = 0; i < buffers.size(); ++i) {
                buffers[i]->freeMem();
                delete buffers[i];
            }
            buffers.clear();
            kernel.freeMem();
            queue.freeMem();
        }
        
        void addParam(int param) {
            kernelLaunch.addArg(param);
        }
        
        Buffer* addParam(void* hostSrc, size_t bytes) {
            Buffer* buf = new Buffer;
            buf->init(queue.get(), context, hostSrc, bytes);
            buffers.push_back(buf);
            kernelLaunch.addArg(*buf);
            return buf;
        }
        
        void setKernel(const std::string& kernelName){
            kernel.init(kernelName.c_str(), program);
            kernelLaunch.init(&kernel);
        }
        void wait(){
            kernelLaunch.wait(queue.get(), context);
        }
        
        
        ~Device(){
            freeMem();
        }
        
        DeviceType getType() const { return deviceType; }
        VendorType getVendor() const { return vendorType; }
        Platform getPlatform() const { return platform; }
        Context getContext() const { return context; }
        DeviceID getID() const { return device; }
        Program getProgram() const { return program; }
        Kernel getKernel() const { return kernel; }
        GPU_QUEUE getQueue() const { return queue.get();}
        size_t getLocalMemSize() const { return maxLocalMemSize; }
        size_t getThreadsPerBlock() const { return maxThreadsPerBlock; }
    private:
        DeviceType deviceType;
        VendorType vendorType;
        
        Platform platform;
        DeviceID device;
        Context context;
        Program program;
        Kernel kernel;
        Queue queue;
        KernelLaunch kernelLaunch;
        //
        size_t maxLocalMemSize;
        size_t maxThreadsPerBlock;
        //
        std::vector<Buffer*> buffers;
    };
}