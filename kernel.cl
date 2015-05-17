#ifdef __OPENCL_VERSION__
    #define KERNEL __kernel
    #define GLOBAL __global
    #define DEVICE
    #define SHARED __local
#endif

#ifdef __CUDACC__  
    #define KERNEL extern "C" __global__
    #define GLOBAL
    #define DEVICE __device__
    #define SHARED __shared__
#endif

#if (!defined __OPENCL_VERSION__) && (!defined __CUDACC__)
    #define KERNEL inline
    #define GLOBAL
    #define DEVICE inline
    #define SHARED
#endif

DEVICE
int getGlobalId() {
#ifdef __OPENCL_VERSION__
    return get_global_id(0);
#endif //__OPENCL_VERSION__
#ifdef __CUDACC__
    return blockIdx.x*blockDim.x + threadIdx.x;
#endif //__CUDACC__
#ifdef __NATIVE__
    extern int threadIdx;
    return threadIdx;
#endif
}

KERNEL
void vecAdd(GLOBAL int *a, GLOBAL int *b, GLOBAL int *c, const unsigned int n) {
    //Get our global thread ID
    int id = getGlobalId();

    //Make sure we do not go out of bounds
    for (int i = 0; i < n; ++i) {
        if (id < n) {
            int ai = a[id] + 1;
            int bi = b[id];
            c[id] = ai + bi;
        }
    }
    
}