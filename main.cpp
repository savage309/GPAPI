
#include <memory>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>
#include <future>
#include <ctime>

//#define TARGET_CUDA
#define TARGET_OPENCL

#include "buffer.h"
#include "opencl_misc.h"
#include "cuda_misc.h"
#include "kernel.h"

using namespace GPAPI;


std::string getProgramSource(const std::string& path) {
    std::ifstream programSource(path);
    if (!programSource.good()) {
        printLog(LogType::Error, "program source not found\n");
        exit(EXIT_FAILURE);
    }
    std::string source((std::istreambuf_iterator<char>(programSource)),std::istreambuf_iterator<char>());
    return std::move(source);
}

enum class API {
    OpenCL, CUDA, Native
};

struct KernelLaunch {
    
};

struct Device {
    std::string name;
    
    GPU_PLATFORM platform;
    GPU_DEVICE device;
    GPU_CONTEXT context;
    GPU_PROGRAM program;
    GPU_QUEUE queue;
    Kernel kernel;

    //
    Buffer d_a;
    Buffer d_b;
    Buffer d_c;
    void init(GPU_PLATFORM platformId, GPU_DEVICE deviceId, std::string nameId, GPU_CONTEXT contextId, GPU_PROGRAM programId) {
        platform = platformId;
        device = deviceId;
        context = contextId;
        program = programId;
        name = nameId;

        GPU_RESULT err = GPU_SUCCESS;
#ifdef TARGET_OPENCL
        cl_command_queue_properties queueFlags;
        err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queueFlags), &queueFlags, NULL);
        CHECK_ERROR(err);
        
        queueFlags &= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        
        queue = clCreateCommandQueue(context, device, 0, &err);
        CHECK_ERROR(err);
#endif
    }
    
    void launchKernel() {
        GPU_RESULT err = GPU_SUCCESS;
        
        printLog(LogType::Info, "kernel %s for device %s launched\n", "vecAdd", name.c_str());
        
        kernel.init("vecAdd", program);
        
        int NUM_ELEMENTS = 32*1024;
        
        std::unique_ptr<int[]> h_a (new int[NUM_ELEMENTS]);
        std::unique_ptr<int[]> h_b (new int[NUM_ELEMENTS]);
        std::unique_ptr<int[]> h_c (new int[NUM_ELEMENTS]);
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            h_a[i] = i;
            h_b[i] = i * 2;
        }
        
        const size_t bytes = NUM_ELEMENTS * sizeof(int);
        
        d_a.init(queue, context, h_a.get(), bytes);
        d_b.init(queue, context, h_b.get(), bytes);
        d_c.init(queue, context, NULL, bytes);
        
        size_t globalSize, localSize;
        // Number of work items in each local work group
        localSize = 64;
        
        // Number of total work items - localSize must be devisor
        globalSize = ceil(NUM_ELEMENTS/(float)localSize)*localSize;
        
        // Create the input and output arrays in device memory for our calculation
#ifdef TARGET_OPENCL
        // Set the arguments to our compute kernel
        err = clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), d_a.get());
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), d_b.get());
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel.get(), 2, sizeof(cl_mem), d_c.get());
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel.get(), 3, sizeof(unsigned int), &NUM_ELEMENTS);
        CHECK_ERROR(err);

       
        // Execute the kernel over the entire range of the data set
        cl_event waitEvent;
        err = clEnqueueNDRangeKernel(queue, kernel.get(), 1, NULL, &globalSize, &localSize,
                                     0, NULL, &waitEvent);
        CHECK_ERROR(err);
        printLog(LogType::Info, "enqueue from device %s\n", name.c_str());
        // Wait for the command queue to get serviced before reading back results
        clWaitForEvents(1, &waitEvent);
        err = clFinish(queue);
        CHECK_ERROR(err);
        printLog(LogType::Info, "finish from device %s\n", name.c_str());
        // Read the results from the device
        err = clEnqueueReadBuffer(queue, *(cl_mem*)d_c.get(), GPU_TRUE, 0,
                            bytes, h_c.get(), 0, NULL, NULL );
        CHECK_ERROR(err);
        
        printLog(LogType::Info, "results from device %s\n", name.c_str());
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
           //printf ("%i ", h_c[i]);
        }
        fflush(stdout);
#endif //TARGET_OPENCL
    }

};


template <typename PLATFORMS, typename DEVICES, typename NAMES, typename CONTEXTS, typename PROGRAMS>
inline void initGPU(PLATFORMS& platformIds, DEVICES& deviceIds, NAMES& deviceNames, CONTEXTS& contextIds, PROGRAMS& programIds, const::std::string& source) {
#ifdef TARGET_OPENCL
    initOpenCL(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
#endif //TARGET_OPENCL
    
#ifdef TARGET_CUDA
    initCUDA(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
#endif
}


int main(int argc, const char * argv[]) {
    using namespace std;
    
    GPU_RESULT err = GPU_SUCCESS;
    
    std::vector<GPU_PLATFORM> platformIds;
    std::vector<GPU_DEVICE> deviceIds;
    std::vector<GPU_CONTEXT> contextIds;
    std::vector<GPU_PROGRAM> programIds;
    std::vector<std::string> deviceNames;

    std::string source = getProgramSource("/Developer/git/opencl/opencl/kernel.cl");
    
    initGPU(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
    
    std::unique_ptr<Device[]> devices(new Device[deviceIds.size()]);
    
    std::vector<std::thread> threads;
    for (int i = 0; i < deviceIds.size(); ++i) {
        devices[i].init(platformIds[i], deviceIds[i], deviceNames[i], contextIds[i], programIds[i]);
        threads.push_back(std::thread(&Device::launchKernel, &devices[i]));
    }
    
    for (auto& t: threads)
        t.join();
    
    CHECK_ERROR(err);
    
    return 0;
}
