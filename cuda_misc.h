//
//  cuda_misc.h
//  opencl
//
//  Created by savage309 on 17.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef opencl_cuda_misc_h
#define opencl_cuda_misc_h

#if !(defined TARGET_OPENCL) && !(defined TARGET_CUDA) && !(defined TARGET_NATIVE)
#error GPAPI::opencl_misc needs one of CUDA, OpenCL or NATIVE targets to be defined
#endif

#ifdef TARGET_CUDA
#include <memory>

#include "common.h"

namespace GPAPI {
    template <typename PLATFORMS, typename DEVICES, typename NAMES, typename CONTEXTS, typename PROGRAMS>
    void initCUDA(PLATFORMS& platformIds, DEVICES& deviceIds, NAMES& deviceNames, CONTEXTS& contextIds, PROGRAMS& programIds, const::std::string& source) {
        
        GPU_RESULT err = GPU_SUCCESS;
        err = cuInit(0);
        CHECK_ERROR(err);
        int count = 0;
        err = cuDeviceGetCount(&count);
        CHECK_ERROR(err);
        
        for (int i = 0; i < count; ++i) {
            GPU_DEVICE device;
            err = cuDeviceGet(&device, i);
            CHECK_ERROR(err);
            deviceIds.push_back(device);
            
            char* buffer[1024];
            err = cuDeviceGetName(	(char*)buffer,
                                         (int)sizeof(buffer),
                                         device
                                     );
            CHECK_ERROR(err);
            deviceNames.push_back(std::string((char*)buffer));
            
            printLog(LogType::Info, "found device '%i' = %s\n", i, buffer);
            
            GPU_CONTEXT pctx;
            err = cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, device);
            CHECK_ERROR(err);
            
            contextIds.push_back(pctx);
            
        }
        nvrtcResult nvRes;
        nvrtcProgram program;
        nvRes = nvrtcCreateProgram(&program, source.c_str(), "compiled_kernel", 0, NULL, NULL);
        CHECK_ERROR(nvRes);
        const char* options[3] = {"--gpu-architecture=compute_20","--maxrregcount=64","--use_fast_math"};
        nvRes = nvrtcCompileProgram(program, 3, options);
        
        if (nvRes != NVRTC_SUCCESS) {
            size_t programLogSize;
            nvRes = nvrtcGetProgramLogSize(program, &programLogSize);
            CHECK_ERROR(nvRes);
            std::unique_ptr<char[]> log(new char[programLogSize + 1]);
            
            nvRes = nvrtcGetProgramLog(program, log.get());
            CHECK_ERROR(nvRes);
            printLog(LogType::Error, "%s", log.get());
        }
        
        size_t ptxSize;
        nvRes = nvrtcGetPTXSize(program, &ptxSize);
        CHECK_ERROR(nvRes);
        
        std::unique_ptr<char[]> ptx(new char[ptxSize + 1]);
        nvRes = nvrtcGetPTX(program, ptx.get());
        
        const size_t JIT_NUM_OPTIONS = 8;
        const size_t JIT_BUFFER_SIZE_IN_BYTES = 1024;
        char logBuffer[JIT_BUFFER_SIZE_IN_BYTES];
        char errorBuffer[JIT_BUFFER_SIZE_IN_BYTES];
        
        CUjit_option jitOptions[JIT_NUM_OPTIONS];
        int optionsCounter = 0;
        jitOptions[optionsCounter++] = CU_JIT_MAX_REGISTERS;
        jitOptions[optionsCounter++] = CU_JIT_OPTIMIZATION_LEVEL;
        jitOptions[optionsCounter++] = CU_JIT_TARGET_FROM_CUCONTEXT;
        jitOptions[optionsCounter++] = CU_JIT_FALLBACK_STRATEGY;
        jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER;
        jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER;
        jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        void* jitValues[JIT_NUM_OPTIONS];
        const int maxRegCount = 63;
        int valuesCounter = 0;
        jitValues[valuesCounter++] = (void*)maxRegCount;
        const int optimizationLevel = 4;
        jitValues[valuesCounter++] = (void*)optimizationLevel;
        const int dummy = 0;
        jitValues[valuesCounter++] = (void*)dummy;
        const CUjit_fallback_enum fallbackStrategy = CU_PREFER_PTX;
        jitValues[valuesCounter++] = (void*)fallbackStrategy;
        jitValues[valuesCounter++] = (void*)logBuffer;
        const int logBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
        jitValues[valuesCounter++] = (void*)logBufferSize;
        jitValues[valuesCounter++] = (void*)errorBuffer;
        const int errorBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
        jitValues[valuesCounter++] = (void*)errorBufferSize;
       
        for (int i = 0; i < deviceIds.size(); ++i) {
            GPU_PROGRAM program;
            err = cuModuleLoadDataEx(&program, ptx.get(), JIT_NUM_OPTIONS, jitOptions, jitValues);
            CHECK_ERROR(err);
            programIds.push_back(program);
            printLog(LogType::Info, "program for device %i compiled\n", i);
        }
    }
}

#endif //TARGET_CUDA
#endif
