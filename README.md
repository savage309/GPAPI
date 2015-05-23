# What is GPAPI :

![alt tag](https://imgs.xkcd.com/comics/standards.png)

So, OpenCL is awesome in terms it provides a way to write once and run everywhere.
However it has some issues - mostly it lacks decent tools for debugging and it does not run at full speed on some nVidia GPUs (for which one would prefer to use CUDA).

GPAPI targets those by providing a way to write code once and to compile and run it using OpenCL, CUDA or C++(98).
Thus you can have the portability of OpenCL, the speed of CUDA and the debugging tools of C++ compilers.

Of course, it comes at a price. And this price is that you can use only the common features along of those 3 languages.

# Sample use case :
######kernel.cl
```
KERNEL
void hello_world() {
    int id = globalID();
    printf("hello from %i\n", id);
}
```


######main.cpp
```
vector<Device> devices;
string source = getProgramSource("kernel.cl");
initGPAPI(devices, source, initParams); //stores handle to all the devices in the system and compiles the source for each one of them on the fly
auto globalSize = 64; auto localSize = 1;
for (auto& device : devices) {
    device.setKernel("hello_world"); //set which kernel from the source we want to call
    device.launchKernel(globalSize, localSize); //launches the kernel
    device.wait(); //wait for the result
}
```

GPAPI (General Purpose API) is designed to be as-simple-as-possible (in contrast to as-powerful-as-possible). The current version lacks textures support, complex memory management and others, 
but can be used for many GPGPU apps.

This is the work-in-progress repo for GPAPI and it still changes as time goes by, but if you ignore that it is in somewhat usable state (and after all, GPAPI does not really targets to be yet-another-GPU-target-languages. It just shows how close those are and how in fact we could use any one of them, instead creating more).

Proper documentation, how-to-build and how-to-use should come in the future.
