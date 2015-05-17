

#include <thread>

#include "configure.h"
#include "gpapi.h"

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


struct DeviceVecAdd : Device {
    //
    Buffer d_a;
    Buffer d_b;
    Buffer d_c;
    
    void launchKernel() {
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
        
        d_a.init(queue.get(), context, h_a.get(), bytes);
        d_b.init(queue.get(), context, h_b.get(), bytes);
        d_c.init(queue.get(), context, NULL, bytes);
        
        size_t globalSize, localSize;
        // Number of work items in each local work group
        localSize = 64;
        
        // Number of total work items - localSize must be devisor
        globalSize = ceil(NUM_ELEMENTS/(float)localSize)*localSize;
        
        // Create the input and output arrays in device memory for our calculation
        // Set the arguments to our compute kernel
       
        KernelLaunch kernelLaunch(kernel);
        kernelLaunch.addArg(d_a);
        kernelLaunch.addArg(d_b);
        kernelLaunch.addArg(d_c);
        kernelLaunch.addArg(NUM_ELEMENTS);
        
        // Execute the kernel over the entire range of the data set
        kernelLaunch.run(queue, context, globalSize, localSize);
        printLog(LogType::Info, "enqueue from device %s\n", name.c_str());
        // Wait for the command queue to get serviced before reading back results
        kernelLaunch.wait(queue, context);
        printLog(LogType::Info, "finish from device %s\n", name.c_str());
        // Read the results from the device
        
        d_c.download(queue.get(), context, h_c.get(), bytes);
        
        printLog(LogType::Info, "results from device %s\n", name.c_str());
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
           printf ("%i ", h_c[i]);
        }
    }

};


int main(int argc, const char * argv[]) {
    using namespace std;
    
    std::vector<Platform> platformIds;
    std::vector<DeviceID> deviceIds;
    std::vector<Context> contextIds;
    std::vector<Program> programIds;
    std::vector<std::string> deviceNames;

    std::string source = getProgramSource("/Developer/git/opencl/opencl/kernel.cl");
    
    initGPAPI(platformIds, deviceIds, deviceNames, contextIds, programIds, source);
    
    std::unique_ptr<DeviceVecAdd[]> devices(new DeviceVecAdd[deviceIds.size()]);
    
    std::vector<std::thread> threads;
    for (int i = 0; i < deviceIds.size(); ++i) {
        devices[i].init(platformIds[0], deviceIds[i], deviceNames[i], contextIds[i], programIds[i]);
        threads.push_back(std::thread(&DeviceVecAdd::launchKernel, &devices[i]));
    }
    
    for (auto& t: threads)
        t.join();
    
    freeGPAPI(platformIds, deviceIds, deviceNames, contextIds, programIds);
    
    return 0;
}
