
#include <thread>

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

int main(int argc, const char * argv[]) {
    using namespace std;

    //load kernel source from a file
    std::string source = getProgramSource("/Developer/git/opencl/opencl/kernel.cl");
    //devices will hold handles to all available GPAPI devices
    std::vector<Device*> devices;
    //filter to get only the devices we want
    InitParams initParams;
    initParams.nvidia = 1; //turn off all devices from nvidia
    initParams.intel = 0; //turn off all intel gpus
    initParams.amd = 0; //turn off all devices from amd
    
    //call initGPAPI to init the devices
    initGPAPI<Device>(devices, source, initParams);
    
    //now prepare some host buffers that will be transfered to the devices
    int NUM_ELEMENTS = 1024;
    std::unique_ptr<int[]> h_a (new int[NUM_ELEMENTS]);
    std::unique_ptr<int[]> h_b (new int[NUM_ELEMENTS]);
    std::unique_ptr<int[]> h_c (new int[NUM_ELEMENTS]);
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    const size_t bytes = NUM_ELEMENTS * sizeof(int);
    //calculate global and local size
    size_t globalSize, localSize;
    // Number of work items in each local work group
    localSize = 64;
    
    // Number of total work items - localSize must be devisor
    globalSize = ceil(NUM_ELEMENTS/(float)localSize)*localSize;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < devices.size(); ++i) {
        Device& device = *(devices[i]);
        //set the kernel name we want to call
        device.setKernel("vecAdd");
        //our kernel has 4 args - 2 input buffers, 1 ouptut buffer and a size
        //set those args
        device.addParam(h_a.get(), bytes);
        device.addParam(h_b.get(), bytes);
        Buffer* result = device.addParam(NULL, bytes);
        device.addParam(NUM_ELEMENTS);
        
        //launch the kernel
        device.launchKernel(globalSize, localSize);
        //wait for the result
        device.wait();
        //copy back the data of the result from the device to the host
        result->download(device.queue.get(), device.context, h_c.get(), bytes);
        
        for (int i = 0; i < NUM_ELEMENTS; ++i) {
            printf ("%i ", h_c[i]);
        }
        
        //and clean up
        device.freeMem();
    }
    
    freeGPAPI(devices);
    
    return 0;
}
