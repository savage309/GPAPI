

#ifndef opencl_common_h
#define opencl_common_h

#if !(defined __GPAPI_H__) && !(defined __GPAPI_NATIVE_MISC_H__)
#   error For GPAPI you need only to include gpapi.h
#endif

#ifdef TARGET_OPENCL
#   if (defined TRAGET_CUDA) || (defined TARGET_NATIVE)
#       error Only one of OpenCL, CUDA, Native targets can be defined at a time
#   endif

#ifdef __APPLE__
#   include "OpenCL/cl.h"
#else
#   include "cl.h"
#endif
#define GPU_PLATFORM cl_platform_id
#define GPU_SUCCESS CL_SUCCESS
#define GPU_INT cl_int
#define GPU_RESULT cl_int
#define GPU_UINT cl_uint
#define GPU_TRUE CL_TRUE
#define GPU_FALSE CL_FALSE
#define GPU_QUEUE cl_command_queue
#define GPU_CONTEXT cl_context
#define GPU_DEVICE cl_device_id
#define GPU_BUFFER cl_mem
#define GPU_PROGRAM cl_program
#define GPU_KERNEL cl_kernel
#define GPU_STREAM cl_stream
#endif

#ifdef TARGET_CUDA
#   if (defined TARGET_OPENCL) || (defined TARGET_NATIVE)
#       error Only one of OpenCL, CUDA, Native targets can be defined at a time
#   endif

#include "cuda.h"
#include "../nvrtc/include/nvrtc.h"

#define GPU_PLATFORM int
#define GPU_TRUE true
#define GPU_FALSE false
#define GPU_INT int
#define GPU_UINT unsigned int
#define GPU_RESULT CUresult
#define GPU_CONTEXT CUcontext
#define GPU_DEVICE CUdevice
#define GPU_BUFFER CUdeviceptr
#define GPU_RESULT CUresult
#define GPU_SUCCESS CUDA_SUCCESS
#define GPU_PROGRAM CUmodule
#define GPU_KERNEL CUfunction
#define GPU_STREAM CUstream
#define GPU_QUEUE void*
#endif

#ifdef TARGET_NATIVE
#   if (defined TARGET_OPENCL) || (defined TARGET_CUDA)
#       error Only one of OpenCL, CUDA, Native targets can be defined at a time
#   endif
#
#define GPU_CONTEXT int
#define GPU_QUEUE int
#define GPU_RESULT int
#define GPU_SUCCESS 0
#define GPU_DEVICE int
#define GPU_KERNEL int
#define GPU_PROGRAM int
#define GPU_PLATFORM int
#endif

#define Platform GPU_PLATFORM
#define DeviceID GPU_DEVICE
#define Context GPU_CONTEXT
#define Program GPU_PROGRAM

#include <memory>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>

enum class LogType { Info = 0, Warning, Error };

inline
void printLog(LogType priority, const char *format, ...) {
    static const LogType CURRENT_LOG_TYPE = LogType::Info;
    char s[512];
    
    time_t t = time(NULL);
    struct tm * p = localtime(&t);
    strftime(s, 512, "[%H:%M:%S] ", p);
    
    printf("%s", s);
    switch (priority) {
        case LogType::Info:
            printf("Info: ");
            break;
        case LogType::Warning:
            printf("Warning: ");
            break;
        case LogType::Error:
            printf("Error: ");
            break;
        default:
            break;
    }
    
    va_list args;
    va_start(args, format);
    
    if(priority >= CURRENT_LOG_TYPE)
        vprintf(format, args);
    
    va_end(args);
}

template<typename T>
inline
void __checkError(T error, const char* file, int line) {
    if (error != 0) {
        printLog(LogType::Error, "error %i in file %s, line %i", error, file, line);
        exit(error);
    }
}

#define CHECK_ERROR(X) __checkError(X, __FILE__, __LINE__)

inline
void pushContext(GPU_CONTEXT context) {
#ifdef TARGET_CUDA
    GPU_RESULT err = cuCtxPushCurrent(context);
    CHECK_ERROR(err);
#endif
}

inline
void popContext(GPU_CONTEXT context) {
#ifdef TARGET_CUDA
    GPU_RESULT err = cuCtxPopCurrent(&context);
    CHECK_ERROR(err);
#endif
}


#endif
