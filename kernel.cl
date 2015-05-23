
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common GPAPI stuff
//////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __OPENCL_VERSION__
    #define KERNEL __kernel
    #define GLOBAL __global
    #define DEVICE
    #define SHARED __local
    #define FLOAT4 float4
    #define RESTRICT restrict
#endif

#ifdef __CUDACC__  
    #define KERNEL extern "C" __global__
    #define GLOBAL
    #define DEVICE __device__
    #define SHARED __shared__
    #define FLOAT4 make_float4
    #define RESTRICT __restrict__
#endif

#if (!defined __OPENCL_VERSION__) && (!defined __CUDACC__)
    #include <cmath>
    using std::min;
    using std::max;
    using std::sqrt;
    #define KERNEL inline
    #define GLOBAL
    #define DEVICE inline
    #define SHARED
    #define FLOAT4 float4
    #define RESTRICT
#endif

#if !defined(__CUDACC__) && !defined(__OPENCL_VERSION__)
struct float4 {
    float x, y, z, w;
    DEVICE float4() {}
    DEVICE float4(float x, float y, float z, float w):x(x),y(y),z(z),w(w){}
};
#endif

#if !defined(__OPENCL_VERSION__)
DEVICE float dot( float4 a, float4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
DEVICE float length( float4 v ) { return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w ); }
DEVICE float4 normalize( float4 v ){ float l = 1.0f / sqrtf( dot(v,v) ); return FLOAT4( v.x*l, v.y*l, v.z*l, v.w*l ); }
DEVICE float4 cross( float4 a, float4 b ) { return FLOAT4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f ); }
DEVICE float clamp(float f, float a, float b) {  return max(a, min(f, b)); }
DEVICE float4 operator*(const float4 &a, const float b ) { return FLOAT4( a.x*b, a.y*b, a.z*b, a.w*b); }
DEVICE float4 operator/(const float4 &a, const float b ) { return FLOAT4( a.x/b, a.y/b, a.z/b, a.w/b );}
DEVICE float4 operator*(const float b, const float4 &a) { return FLOAT4( a.x*b, a.y*b, a.z*b, a.w*b ); }
DEVICE float4 operator+(const float4 &a, const float4 &b ) { return FLOAT4( a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w ); }
DEVICE float4 operator-(const float4 &a, const float4 &b ) { return FLOAT4( a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w );}
DEVICE float4 operator*(const float4 &a, const float4 &b ) { return FLOAT4( a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w ); }
DEVICE float4 operator-(const float4 &b) { return FLOAT4( -b.x, -b.y, -b.z, -b.w );}
DEVICE float4 min( const float4 &a, const float4 &b ) { return FLOAT4( min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w) );}
DEVICE float4 max( const float4 &a, const float4 &b ) { return FLOAT4( max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w) ); }
DEVICE float4& operator*=(float4 &a, const float4 &b ) { a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w; return a; }
DEVICE float4& operator*=(float4 &a, const float &b ) { a.x*=b; a.y*=b; a.z*=b; a.w*=b; return a; }
DEVICE float4& operator+=(float4 &a, const float4 &b ) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; return a; }
DEVICE float4& operator-=(float4 &a, const float4 &b ) { a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w; return a; }
DEVICE float4& operator/=(float4 &a, const float &b ) { a.x/=b; a.y/=b; a.z/=b; a.w/=b; return a; }
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////

KERNEL
void vecAdd(GLOBAL int * RESTRICT a,
            GLOBAL int * RESTRICT b,
            GLOBAL int * RESTRICT c,
            unsigned int n)
{
    float4 af = FLOAT4(0, 0, 0, 0);
    float4 bf = FLOAT4(1, 1, 1, 1);
    float4 resf = af * bf;
    
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