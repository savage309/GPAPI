#pragma once

#define __GPAPI_H__

#include "configure.h"

#if !(defined TARGET_OPENCL) && !(defined TARGET_CUDA) && !(defined TARGET_NATIVE)
#   error GPAPI::opencl_misc needs one of CUDA, OpenCL or NATIVE targets to be defined
#endif

#include <algorithm>

#include "buffer.h"
#include "opencl_misc.h"
#include "cuda_misc.h"
#include "queue.h"
#include "kernel.h"
#include "kernel_launch.h"
#include "device.h"
#include "native_misc.h"

namespace GPAPI {
    
    
    template <typename DEVICE>
    inline void initGPAPI(std::vector<DEVICE*>& devices, const::std::string& source, InitParams initParams = InitParams()) {
        std::vector<Platform> platformIds;
        std::vector<DeviceID> deviceIds;
        std::vector<Context> contextIds;
        std::vector<Program> programIds;
        std::vector<std::string> deviceNames;
        std::vector<size_t> localMemSize;
        std::vector<size_t> threadsPerBlock;
        
#ifdef TARGET_OPENCL
        initOpenCL(platformIds, deviceIds, deviceNames, contextIds, programIds, source, localMemSize, threadsPerBlock, initParams);
#endif //TARGET_OPENCL
        
#ifdef TARGET_CUDA
        initCUDA(platformIds, deviceIds, deviceNames, contextIds, programIds, source, localMemSize, threadsPerBlock, initParams);
#endif
#ifdef TARGET_NATIVE
        platformIds.push_back(0);
        deviceIds.push_back(0);
        deviceNames.push_back("NATIVE");
        contextIds.push_back(0);
        programIds.push_back(0);
        threadsPerBlock.push_back(1);
        localMemSize.push_back(1024 * 1024);
        printLog(LogTypeInfo, "found device 0 = \"Native\", sharedMem=%i, threadsPerBlock=%i\n", (int)localMemSize[0], (int)threadsPerBlock[0]);

#endif
        for (int i = 0; i < deviceIds.size(); ++i) {
            DEVICE* d = new DEVICE;
            d->init(platformIds[0], deviceIds[i], deviceNames[i], contextIds[i], programIds[i], getVendorType(deviceNames[i]), getDeviceType(deviceNames[i]), localMemSize[i], threadsPerBlock[i]);
            devices.push_back(d);
        }

        if (!devices.size()) {
            printLog(LogTypeWarning, "No valid devices found\n");
        }
    }
    
    
    template <typename DEVICE>
    inline void freeGPAPI(std::vector<DEVICE*> devices) {
        GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
        for (int i = 0; i < devices.size(); ++i) {
            devices[i]->freeMem();
            
            err = clReleaseDevice(devices[i]->getID());
            CHECK_ERROR(err);
            err = clReleaseContext(devices[i]->getContext());
            CHECK_ERROR(err);
            err = clReleaseProgram(devices[i]->getProgram());
            CHECK_ERROR(err);
        }
        
#endif
#ifdef TARGET_CUDA
        for (int i = 0; i < devices.size(); ++i) {
            devices[i]->freeMem();

            err = cuModuleUnload(devices[i]->getProgram());
            CHECK_ERROR(err);
            err = cuCtxDestroy(devices[i]->getContext());
            CHECK_ERROR(err);
        }
#endif
        for (int i = 0; i < devices.size(); ++i) {
            delete devices[i];
        }
        CHECK_ERROR(err);
    }
}