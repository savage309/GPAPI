#pragma once

namespace GPAPI {
    /*! \class InitParams
     \brief Class used to specialize which devices the user want to init when initGPAPI is called
     
     initGPAPI queries for all available devices, and inits only those, for which InitParams::isActive returns 1
     InitParams can filter devices by vendor type (nvidia, amd, intel, other) and by type (cpu, gpu, accel, other)
     Example:
     InitParams params; //the default ctor sets all devices to be used
     params.intel = 0; //turn off all devices from intel
     params.nvidia = 0; //turn off all devices from nvidia
     params.amd = 0; //turn off all devices from amd
     params.amd.cpu = 1; //turn on only the cpu devices from amd
     //as a result, only CPU devices from AMD will be used
     */
    struct InitParams {
        struct VendorParams {
            /** Device type enumeration
             *  CPU, GPU and Accel devices match the OpenCL enumeration. UnkownDevice is extension, used to specify the device type when TARGET_NATIVE is defined
             */
            enum DeviceType { CPU, GPU, Accel, UnkownDevice };
            /** Device vendor enumeration
             *  UnknownVendor is used to specify the device vendor when TARGET_NATIVE is defined
             */
            enum VendorType { Intel, NVidia, AMD, UnknownVendor };
            //! Inits all device types to be turned on by default
            VendorParams() {
                setMask(-1, GPU);
                setMask(-1, CPU);
                setMask(-1, Accel);
                setMask(-1, UnkownDevice);
            }
            /*! Sets if this vendor should be used at all
            \param i if the value is 0, this vendor devices will not be used. If it is !=0, this vendor devices will be used
            */
            unsigned operator=(unsigned i) {
                if (i) gpu = cpu = accel = other = unsigned(-1);
                else gpu = cpu = accel = other = 0;
                return i;
            }
            /*! Sets which devices from the vendor should be used
             \param mask bit-mask. If the i-th bit is raised, deviced the i-th device with \type will be used. If the i-th bit is not raised, the i-th device will not be used
             \param type specifies if we are setting bit-mask for cpu, gpu, accel or unkowndevice
             */
            void setMask(unsigned mask, DeviceType type) {
                switch(type) {
                    case CPU:
                        cpu = mask;
                        break;
                    case GPU:
                        gpu = mask;
                        break;
                    case Accel:
                        accel = mask;
                        break;
                    case UnkownDevice:
                        other = mask;
                        break;
                }
            }
            /*! Sets if this vendor should be used at all
             \return 1 if the deviceType at position index should be used
             */
            int isActive(DeviceType deviceType, unsigned index) {
                switch(deviceType) {
                    case CPU:
                        return cpu & (1 << index);
                        break;
                    case GPU:
                        return gpu & (1 << index);
                        break;
                    case Accel:
                        return accel & (1 << index);
                        break;
                    case UnkownDevice:
                        return other & (1 << index);
                        break;
                }
                return 0;
            }
            unsigned gpu;
            unsigned cpu;
            unsigned accel;
            unsigned other;
        };
        
        VendorParams intel;
        VendorParams nvidia;
        VendorParams amd;
        VendorParams other;
        int isActive(InitParams::VendorParams::VendorType vendor,
                     InitParams::VendorParams::DeviceType device,
                     unsigned i) {
            switch (vendor) {
                case VendorParams::Intel:
                    return intel.isActive(device, i);
                    break;
                case VendorParams::NVidia:
                    return nvidia.isActive(device, i);
                    break;
                case VendorParams::AMD:
                    return amd.isActive(device, i);
                    break;
                case VendorParams::UnknownVendor:
                    return other.isActive(device, i);
                    break;
            }
        }
        unsigned operator=(unsigned i) {
            intel = nvidia = amd = other = i;
            return i;
        }	
    };

}
