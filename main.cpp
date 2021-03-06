#include "gpapi.h"

using namespace GPAPI;

/*! \return Content of the file, that has absolute path 'path' */
std::string getProgramSource(const std::string& path) {
	std::ifstream programSource(path.c_str());
	if (!programSource.good()) printLog(LogTypeError, "program source not found\n");
	return std::string((std::istreambuf_iterator <char>(programSource)), std::istreambuf_iterator <char>());
}

int main(int argc, const char *argv[]) {
	using namespace std;

	//load kernel source from a file
	std::string source = getProgramSource("/Developer/git/opencl/opencl/kernel.cl");
	//devices will hold handles to all available GPAPI devices
	std::vector<Device*> devices;
	//filter to get only the devices we want
	InitParams initParams;
	//initParams.nvidia = 1; //turn off all devices from nvidia
	//initParams.intel = 0; //turn off all intel gpus
	//initParams.amd = 0; //turn off all devices from amd

	//call initGPAPI to init the devices
	initGPAPI(devices, source, initParams);

	//now prepare some host buffers that will be transfering to the devices
	int NUM_ELEMENTS = 1024;
	int *h_a = new int[NUM_ELEMENTS];
	int *h_b = new int[NUM_ELEMENTS];
	int *h_c = new int[NUM_ELEMENTS];
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
	globalSize = ceil(NUM_ELEMENTS / (float)localSize) * localSize;

	for (int i = 0; i < devices.size(); ++i) {
		Device& device = *(devices[i]);
		//set the kernel name we want to call
		device.setKernel("vecAdd");
		//our kernel has 4 args - 2 input buffers, 1 ouptut buffer and a size
		//set those args
		device.addParam(h_a, bytes);
		device.addParam(h_b, bytes);
		Buffer *result = device.addParam(NULL, bytes);
		device.addParam(NUM_ELEMENTS);

		//launch the kernel
		device.launchKernel(globalSize, localSize);
		//wait for the result
		device.wait();
		//copy back the data of the result from the device to the host
		result->download(device.getQueue(), device.getContext(), h_c, bytes);

		for (int i = 0; i < NUM_ELEMENTS; ++i) {
			printf("%i ", h_c[i]);
		}

		//clean up
		device.freeMem();
	}

	freeGPAPI(devices);

	return 0;
}
