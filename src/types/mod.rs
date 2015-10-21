macro_rules! newtype {
    ($($Name:ident : $Type:ty),*) => {
        #[derive(Debug)]
        $(pub struct $Name($Type);)*
    };
    ($($Name:ident : $Type:ty),*,) => {
        newtype!($($Name : $Type),*);
    };
}

#[allow(dead_code)]
pub mod ll {
    use opencl::cl;
    use opencl::error::check;
    use std::ptr;
    use std::mem;
    use std::iter::repeat;

    pub use self::device_type::DeviceType;

    newtype! {
        PlatformId: cl::cl_platform_id,
        DeviceId: cl::cl_device_id,
        Context: cl::cl_context,
        CommandQueue: cl::cl_command_queue,
        Mem: cl::cl_mem,
        Program: cl::cl_program,
        Kernel: cl::cl_kernel,
        Event: cl::cl_event,
        Sampler: cl::cl_sampler,
    }

    pub mod mem_flags {
        use opencl::cl;
        bitflags! {
            flags ClMemFlags: cl::cl_bitfield {
                const ReadWrite = cl::CL_MEM_READ_WRITE,
                const WriteOnly = cl::CL_MEM_WRITE_ONLY,
                const ReadOnly = cl::CL_MEM_READ_ONLY,
                const UseHostPtr = cl::CL_MEM_USE_HOST_PTR,
                const AllocHostPtr = cl::CL_MEM_ALLOC_HOST_PTR,
                const CopyHostPtr = cl::CL_MEM_COPY_HOST_PTR,
            }
        }
    }

    pub mod map_flags {
        use opencl::cl;
        bitflags! {
            flags ClMapFlags: cl::cl_bitfield {
                const Read = cl::CL_MAP_READ,
                const Write = cl::CL_MAP_WRITE,
            }
        }
    }

    pub mod device_type {
        use opencl::cl;
        bitflags! {
            flags DeviceType: cl::cl_bitfield {
                const Default = cl::CL_DEVICE_TYPE_DEFAULT,
                const GPU = cl::CL_DEVICE_TYPE_GPU,
                const CPU = cl::CL_DEVICE_TYPE_CPU,
                const Accelerator = cl::CL_DEVICE_TYPE_ACCELERATOR,
                const All = cl::CL_DEVICE_TYPE_ALL,
            }
        }
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum PlatformInfo {
        Profile = cl::CL_PLATFORM_PROFILE,
        Version = cl::CL_PLATFORM_VERSION,
        Name = cl::CL_PLATFORM_NAME,
        Vendor = cl::CL_PLATFORM_VENDOR,
        Extensions = cl::CL_PLATFORM_EXTENSIONS,
    }

    pub trait DeviceInfo {
        type Result;
        fn get_device_info(self, device: &DeviceId) -> Self::Result;
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum DeviceInfoBool {
        Available = cl::CL_DEVICE_AVAILABLE,
        CompilerAvailable = cl::CL_DEVICE_COMPILER_AVAILABLE,
        EndianLittle = cl::CL_DEVICE_ENDIAN_LITTLE,
        ErrorCorrectionSupport = cl::CL_DEVICE_ERROR_CORRECTION_SUPPORT,
        ImageSupport = cl::CL_DEVICE_IMAGE_SUPPORT,
    }

    impl DeviceInfo for DeviceInfoBool {
        type Result = cl::cl_bool;
        fn get_device_info(self, device: &DeviceId) -> cl::cl_bool {
            unsafe {
                let mut ret = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info, mem::size_of::<cl::cl_bool>() as u64,
                    &mut ret as *mut _ as *mut _, ptr::null_mut());
                check(res, "Failed to get cl_bool parameter");
                ret
            }
        }
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum DeviceInfoString {
        Name = cl::CL_DEVICE_NAME,
        Profile = cl::CL_DEVICE_PROFILE,
        Vendor = cl::CL_DEVICE_VENDOR,
        DeviceVersion = cl::CL_DEVICE_VERSION,
        DriverVersion = cl::CL_DRIVER_VERSION,
        Extensions =  cl::CL_DEVICE_EXTENSIONS,
    }

    impl DeviceInfo for DeviceInfoString {
        type Result = String;
        fn get_device_info(self, device: &DeviceId) -> String {
            unsafe {
                let mut str_len = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info, 0, ptr::null_mut(), &mut str_len);
                check(res, "Failed to get length of string result");
                let mut bytes: Vec<_> = repeat(0).take(str_len as usize).collect();
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info, bytes.len() as u64,
                    bytes.as_mut_ptr() as *mut _ as *mut _, ptr::null_mut());
                check(res, "Failed to get string result");
                String::from_utf8_unchecked(bytes)
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct DeviceInfoDeviceType;
    impl DeviceInfo for DeviceInfoDeviceType {
        type Result = DeviceType;
        fn get_device_info(self, device: &DeviceId) -> DeviceType {
            unsafe {
                let mut device_type: cl::cl_device_type = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, cl::CL_DEVICE_TYPE, mem::size_of::<cl::cl_device_type>() as u64,
                    &mut device_type as *mut _ as *mut _, ptr::null_mut());
                check(res, "Failed to get cl_device_type result");
                DeviceType::from_bits_truncate(device_type)
            }
        }
    }

    pub type Result<A> = ::std::result::Result<A, cl::CLStatus>;
    fn try_status(status: cl::CLStatus) -> Result<()> {
        match status {
            cl::CLStatus::CL_SUCCESS => Ok(()),
            other => Err(other),
        }
    }

    pub fn get_platform_ids() -> Vec<PlatformId> {
        unsafe {
            let mut num_platforms = 0;
            let res = cl::ll::clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms);
            check(res, "Failed to get number of platforms");
            let mut ids: Vec<_> = repeat(0 as *mut _).take(num_platforms as usize).collect();
            let res = cl::ll::clGetPlatformIDs(
                ids.len() as u32, ids.as_mut_ptr(), ptr::null_mut());
            check(res, "Failed to get platform IDs");
            ids.iter().map(|ptr| PlatformId(*ptr)).collect()
        }
    }

    pub fn get_platform_info(platform: &PlatformId, info: PlatformInfo) -> String {
        unsafe {
            let mut info_size = 0;
            let res = cl::ll::clGetPlatformInfo(
                platform.0, info as cl::cl_platform_info, 0, ptr::null_mut(), &mut info_size);
            check(res, "Failed to get platform info length");
            let mut bytes: Vec<_> = repeat(0).take(info_size as usize).collect();
            let res = cl::ll::clGetPlatformInfo(
                platform.0, info as cl::cl_platform_info, bytes.len() as u64,
                bytes.as_mut_ptr() as *mut _, ptr::null_mut());
            check(res, "Failed to get platform info string");
            String::from_utf8_unchecked(bytes)
        }
    }

    pub fn get_device_ids(platform: &PlatformId, device_type: DeviceType) -> Vec<DeviceId> {
        unsafe {
            let mut num_devices = 0;
            let res = cl::ll::clGetDeviceIDs(
                platform.0, device_type.bits(), 0, ptr::null_mut(), &mut num_devices);
            check(res, "Failed to get number of platforms");
            let mut ids: Vec<_> = repeat(0 as *mut _).take(num_devices as usize).collect();
            let res = cl::ll::clGetDeviceIDs(
                platform.0, device_type.bits(), ids.len() as u32, ids.as_mut_ptr(),
                ptr::null_mut());
            check(res, "Failed to get platform IDs");
            ids.iter().map(|ptr| DeviceId(*ptr)).collect()
        }
    }

    pub fn get_device_info<T: DeviceInfo>(device: &DeviceId, info: T) -> T::Result {
        info.get_device_info(device)
    }
}

pub mod hl {
    use super::ll;
    newtype! {
        Platform: ll::PlatformId,
        Device: ll::DeviceId,
    }

    pub fn get_platforms() -> Vec<Platform> {
        ll::get_platform_ids().into_iter().map(Platform).collect()
    }

    impl Platform {
        pub fn get_devices(&self) -> Vec<Device> {
            ll::get_device_ids(&self.0, ll::device_type::All).into_iter().map(Device).collect()
        }

        pub fn name(&self) -> String {
            ll::get_platform_info(&self.0, ll::PlatformInfo::Name)
        }

        pub fn version(&self) -> String {
            ll::get_platform_info(&self.0, ll::PlatformInfo::Version)
        }

        pub fn profile(&self) -> String {
            ll::get_platform_info(&self.0, ll::PlatformInfo::Profile)
        }

        pub fn vendor(&self) -> String {
            ll::get_platform_info(&self.0, ll::PlatformInfo::Vendor)
        }

        pub fn extensions(&self) -> String {
            ll::get_platform_info(&self.0, ll::PlatformInfo::Extensions)
        }
    }

    impl Device {
        pub fn name(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::Name)
        }

        pub fn profile(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::Profile)
        }

        pub fn vendor(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::Vendor)
        }

        pub fn device_version(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::DeviceVersion)
        }

        pub fn driver_version(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::DriverVersion)
        }

        pub fn extensions(&self) -> String {
            ll::get_device_info(&self.0, ll::DeviceInfoString::Extensions)
        }

        pub fn device_type(&self) -> ll::DeviceType {
            ll::get_device_info(&self.0, ll::DeviceInfoDeviceType)
        }
    }
}
