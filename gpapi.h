#ifndef __GPAPI_H__
#define __GPAPI_H__

#if !(defined TARGET_OPENCL) && !(defined TARGET_CUDA) && !(defined TARGET_NATIVE)
#   error GPAPI::opencl_misc needs one of CUDA, OpenCL or NATIVE targets to be defined
#endif

#include "buffer.h"
#include "opencl_misc.h"
#include "cuda_misc.h"
#include "queue.h"
#include "kernel.h"
#include "kernel_launch.h"
#include "device.h"
#include "native_misc.h"

namespace GPAPI {
    
    template <typename PLATFORMS, typename DEVICES, typename NAMES, typename CONTEXTS, typename PROGRAMS>
    inline void initGPAPI(PLATFORMS& platformIds, DEVICES& deviceIds, NAMES& deviceNames, CONTEXTS& contextIds, PROGRAMS& programIds, const::std::string& source) {
#ifdef TARGET_OPENCL
        initOpenCL(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
#endif //TARGET_OPENCL
        
#ifdef TARGET_CUDA
        initCUDA(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
#endif
#ifdef TARGET_NATIVE
        platformIds.push_back(0);
        deviceIds.push_back(0);
        deviceNames.push_back("NATIVE");
        contextIds.push_back(0);
        programIds.push_back(0);
#endif
    }
    
    
    template <typename PLATFORMS, typename DEVICES, typename NAMES, typename CONTEXTS, typename PROGRAMS>
    inline void freeGPAPI(PLATFORMS& platformIds, DEVICES& deviceIds, NAMES& deviceNames, CONTEXTS& contextIds, PROGRAMS& programIds) {
        GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
        for (int i = 0; i < deviceIds.size(); ++i) {
            err = clReleaseDevice(deviceIds[i]);
            CHECK_ERROR(err);
        }
        for (int i = 0; i < contextIds.size(); ++i) {
            err = clReleaseContext(contextIds[i]);
            CHECK_ERROR(err);
        }
        for (int i = 0; i < programIds.size(); ++i) {
            err = clReleaseProgram(programIds[i]);
            CHECK_ERROR(err);
        }
#endif
#ifdef TARGET_CUDA
        for (int i = 0; i < programIds.size(); ++i) {
            err = cuModuleUnload(programIds[i]);
            CHECK_ERROR(err);
        }
        for (int i = 0; i < contextIds.size(); ++i) {
            err = cuCtxDestroy(contextIds[i]);
            CHECK_ERROR(err);
        }
#endif
        CHECK_ERROR(err);
    }
}

#endif //__GPAPI_H__
