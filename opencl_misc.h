#pragma once

#include "common.h"

#include "init_params.h"

#ifndef __GPAPI_H__
#   error For GPAPI you need only to include gpapi.h
#endif

namespace GPAPI {
    
    inline
    InitParams::VendorParams::VendorType
    getVendorType(std::string deviceName) {
#ifdef TARGET_NATIVE
        return InitParams::VendorParams::UnknownVendor;
#endif
#ifdef TARGET_CUDA
        return InitParams::VendorParams::NVidia;
#endif
#ifdef TARGET_OPENCL
        std::transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::tolower);
        if (deviceName.find("intel") != std::string::npos) {
            return InitParams::VendorParams::Intel;
        } else if (deviceName.find("geforce") != std::string::npos ||
                   deviceName.find("quadro") != std::string::npos ||
                   deviceName.find("tesla") != std::string::npos) {
            return InitParams::VendorParams::NVidia;
        } else if (deviceName.find("nvidia") != std::string::npos) {
            return InitParams::VendorParams::NVidia;
        } else if (deviceName.find("hd graphics") != std::string::npos) {
            return InitParams::VendorParams::Intel;
        } else if (deviceName.find("amd") != std::string::npos) {
            return InitParams::VendorParams::AMD;
        }
        
        return InitParams::VendorParams::UnknownVendor;
#endif
    }
    
    inline
    InitParams::VendorParams::DeviceType
    getDeviceType(std::string deviceName) {
#ifdef TARGET_NATIVE
        return InitParams::VendorParams::UnkownDevice;
#endif
#ifdef TARGET_CUDA
        return InitParams::VendorParams::GPU;
#endif
#ifdef TARGET_OPENCL
        std::transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::tolower);
        if (deviceName.find("intel(r) core") != std::string::npos) {
            return InitParams::VendorParams::CPU;
        } else if (deviceName.find("geforce") != std::string::npos ||
                   deviceName.find("quadro") != std::string::npos ||
                   deviceName.find("tesla") != std::string::npos) {
            return InitParams::VendorParams::GPU;
        } else if (deviceName.find("nvidia") != std::string::npos) {
            return InitParams::VendorParams::GPU;
        } else if (deviceName.find("hd graphics") != std::string::npos) {
            return InitParams::VendorParams::GPU;
        } else if (deviceName.find("hawaii") != std::string::npos) {
            return InitParams::VendorParams::GPU;
        }
#endif
        return InitParams::VendorParams::UnkownDevice;
    }
    
#ifdef TARGET_OPENCL
    template<typename P>
    void getOCLPlatforms(P& platforms) {
        GPU_RESULT err = GPU_SUCCESS;
        GPU_UINT numPlatforms;
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        CHECK_ERROR(err);
        
        platforms.resize(numPlatforms);
        
        err = clGetPlatformIDs(numPlatforms, &platforms[0], &numPlatforms);
        CHECK_ERROR(err);
        
        for (int i = 0; i < platforms.size(); ++i) {
            char chBuffer[1024];
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
            CHECK_ERROR(err);
            
            printLog(LogType::Info, "found platform '%i' = \"%s\"\n", i, chBuffer);
        }
    }
    
    template <typename D, typename N, typename P>
    void getOCLDevices(D& devices, N& names, const P& platforms, int numMaxDevices = std::numeric_limits<int>::max()) {
        GPU_RESULT err = GPU_SUCCESS;
        
        GPU_INT numDevices = 0;
        for (int i =0; i < platforms.size(); ++i) {
            GPU_UINT devicesCount = 0;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount);
            CHECK_ERROR(err);
            numDevices += devicesCount;
        }
        
        if (numDevices > numMaxDevices) {
            numDevices = numMaxDevices;
        }
        
        devices.resize(numDevices);
        for (int i = 0; i < platforms.size(); ++i) {
            
            GPU_UINT devicesFound = 0;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, &devices[0], &devicesFound);
            CHECK_ERROR(err);
            
            for (int i = 0; i < devicesFound; ++i) {
                char buffer[1024];
                err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
                CHECK_ERROR(err);
                names.push_back(buffer);
                printLog(LogType::Info, "found device '%i' = \"%s\"\n", i, buffer);
            }
        }
    }
    
    template <typename C, typename D, typename P>
    void getOCLContexts(C& contexts, const D& devices, const P& platforms) {
        GPU_RESULT err = GPU_SUCCESS;
        contexts.reserve(devices.size());
        for (int i = 0; i < devices.size(); ++i) {
            cl_context_properties contextProperties[] =
            {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)platforms[0],
                0
            };
            contexts.push_back(clCreateContext(contextProperties, 1, &devices[i], NULL, NULL, &err));
            CHECK_ERROR(err);
        }
    }
    
    template<typename P, typename C>
    void getOCLPrograms(P& programs, const C& contexts, const char* source) {
        size_t lengths[1] = {strlen(source)};
        const char* sources[1] = {source};
        GPU_RESULT err = GPU_SUCCESS;
        for (int i = 0; i < contexts.size(); ++i) {
            programs.push_back(clCreateProgramWithSource(contexts[i],
                                                         1,
                                                         sources,
                                                         lengths,
                                                         &err));
            CHECK_ERROR(err);
        }
    }
    
    template<typename P, typename D>
    void buildOCLPrograms(const P& programs, const D& devices) {
        GPU_RESULT err = GPU_SUCCESS;
        
        for (int i = 0; i < devices.size(); ++i) {
            err = clBuildProgram (programs[i],1,&devices[i],NULL,NULL,NULL);
            if (err != GPU_SUCCESS) {
                
                size_t buildLogSize;
                char buildLog[2048];
                err = clGetProgramBuildInfo (programs[i],devices[i],CL_PROGRAM_BUILD_LOG,2048,buildLog,&buildLogSize);
                CHECK_ERROR(err);
                printLog(LogType::Error, "*** %s", buildLog);
                
            }
            CHECK_ERROR(err);
            printLog(LogType::Info, "program %i compiled successfully\n", i);
        }
    }
    
    template <typename T, typename C>
    void getOCLKernels(T& kernels, const C& programs, const char* kernelName) {
        GPU_RESULT err = GPU_SUCCESS;
        for (int i = 0; i < programs.size(); ++i) {
            kernels.push_back(clCreateKernel(programs[i], kernelName, &err));
            CHECK_ERROR(err);
        }
    }
    
    template <typename PLATFORMS, typename DEVICES, typename NAMES, typename CONTEXTS, typename PROGRAMS>
    void initOpenCL(PLATFORMS& platformIds, DEVICES& deviceIds, NAMES& deviceNames, CONTEXTS& contextIds, PROGRAMS& programIds, const::std::string& source, InitParams initParams) {
        getOCLPlatforms(platformIds);
        getOCLDevices(deviceIds, deviceNames, platformIds);
        
        std::vector<int> indexesToDelete;
        for (int i = 0; i < deviceNames.size(); ++i) {
            InitParams::VendorParams::VendorType vendorType = getVendorType(deviceNames[i]);
            InitParams::VendorParams::DeviceType deviceType = getDeviceType(deviceNames[i]);
            if (!initParams.isActive(vendorType, deviceType, i)) {
                printLog(LogType::Info, "Skipped device named %s\n", deviceNames[i].c_str());
                indexesToDelete.push_back(i);
            }
        }
        
        for (int i = 0; i < indexesToDelete.size(); ++i) {
            deviceIds.erase(deviceIds.begin() + i);
            deviceNames.erase(deviceNames.begin() + i);
        }
        
        getOCLContexts(contextIds, deviceIds, platformIds);
        getOCLPrograms(programIds, contextIds, source.c_str());
        buildOCLPrograms(programIds, deviceIds);
    }
#endif //TARGET_OPENCL
}
