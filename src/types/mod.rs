use opencl::cl;
use num;

pub type Result<A> = ::std::result::Result<A, cl::CLStatus>;
fn check_status(status_int: cl::cl_int) -> Result<()> {
    let status = num::FromPrimitive::from_i32(status_int);
    match status {
        Some(cl::CLStatus::CL_SUCCESS) => Ok(()),
        Some(other) => Err(other),
        None => panic!("Tried to check invalid opencl status! (value was {})", status_int)
    }
}

#[allow(dead_code)]
pub mod ll {
    use opencl::cl;
    use libc;
    use std::ffi::CStr;
    use std::ptr;
    use std::mem;
    use std::iter::repeat;
    use super::check_status;
    use super::Result;

    pub use self::device_type::DeviceType;

    macro_rules! newtype_to_from_raw {
        ($($(#[$Meta:meta])* pub struct $Name:ident($Type:ty));*;) => {
            $(
                $(#[$Meta])*
                pub struct $Name($Type);

                impl $Name {
                    unsafe fn from_raw(raw: $Type) -> Self {
                        $Name(raw)
                    }

                    fn as_raw(&self) -> $Type {
                        self.0
                    }
                }
            )*
        }
    }

    newtype_to_from_raw! {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
        pub struct PlatformId(cl::cl_platform_id);

        #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
        pub struct DeviceId(cl::cl_device_id);

        #[derive(Debug, Copy, Clone)]
        pub struct Context(cl::cl_context);

        #[derive(Debug, Copy, Clone)]
        pub struct CommandQueue(cl::cl_command_queue);

        #[derive(Debug, Copy, Clone)]
        pub struct Mem(cl::cl_mem);

        #[derive(Debug, Copy, Clone)]
        pub struct Program(cl::cl_program);

        #[derive(Debug, Copy, Clone)]
        pub struct Kernel(cl::cl_kernel);

        #[derive(Debug, Copy, Clone)]
        pub struct Event(cl::cl_event);

        #[derive(Debug, Copy, Clone)]
        pub struct Sampler(cl::cl_sampler);
    }

    /// Turns a vector of `u8`s into a Rust string.
    /// Assumes that the vector had a string `s` written to it, with `strlen(s) == buf.len() - 1`
    /// (i.e. the last entry in the vector should be the terminating null character).
    fn string_from_cstring_buf(mut buf: Vec<u8>) -> String {
        let last_char = buf.pop();
        match last_char {
            Some(0u8) => {
                match String::from_utf8(buf) {
                    Ok(string) => string,
                    Err(err) => panic!(
                        "Rascal: Invalid utf8 sequence in buffer! (Error: {:?})", err),
                }
            }
            Some(other) => panic!(
                "Rascal: Last char in buffer wasn't null! (Expected 0, found {})", other),
            None => panic!(
                "Rascal: Tried to turn empty buffer into string! (Expected null character)"),
        }
    }

    pub mod mem_flags {
        use opencl::cl;
        bitflags! {
            flags ClMemFlags: cl::cl_bitfield {
                const READ_WRITE = cl::CL_MEM_READ_WRITE,
                const WRITE_ONLY = cl::CL_MEM_WRITE_ONLY,
                const READ_ONLY = cl::CL_MEM_READ_ONLY,
                const USE_HOST_PTR = cl::CL_MEM_USE_HOST_PTR,
                const ALLOC_HOST_PTR = cl::CL_MEM_ALLOC_HOST_PTR,
                const COPY_HOST_PTR = cl::CL_MEM_COPY_HOST_PTR,
            }
        }
    }

    pub mod map_flags {
        use opencl::cl;
        bitflags! {
            flags ClMapFlags: cl::cl_bitfield {
                const READ = cl::CL_MAP_READ,
                const WRITE = cl::CL_MAP_WRITE,
            }
        }
    }

    pub mod device_type {
        use opencl::cl;
        bitflags! {
            flags DeviceType: cl::cl_bitfield {
                const DEFAULT = cl::CL_DEVICE_TYPE_DEFAULT,
                const GPU = cl::CL_DEVICE_TYPE_GPU,
                const CPU = cl::CL_DEVICE_TYPE_CPU,
                const ACCELERATOR = cl::CL_DEVICE_TYPE_ACCELERATOR,
                const ALL = cl::CL_DEVICE_TYPE_ALL,
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
        type Info;
        fn get_device_info(self, device: DeviceId) -> Result<Self::Info>;
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
        type Info = cl::cl_bool;
        fn get_device_info(self, device: DeviceId) -> Result<cl::cl_bool> {
            unsafe {
                let mut ret = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info,
                    mem::size_of::<cl::cl_bool>() as libc::size_t,
                    &mut ret as *mut _ as *mut _, ptr::null_mut());
                try!(check_status(res));
                Ok(ret)
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
        type Info = String;
        fn get_device_info(self, device: DeviceId) -> Result<String> {
            unsafe {
                let mut str_len = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info, 0, ptr::null_mut(), &mut str_len);
                try!(check_status(res));
                let mut bytes: Vec<_> = repeat(0).take(str_len as usize).collect();
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info, bytes.len() as libc::size_t,
                    bytes.as_mut_ptr() as *mut _ as *mut _, ptr::null_mut());
                try!(check_status(res));
                Ok(string_from_cstring_buf(bytes))
            }
        }
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone)]
    pub enum DeviceInfoClUint {
        MaxClockFrequency = cl::CL_DEVICE_MAX_CLOCK_FREQUENCY,
        MaxComputeUnits = cl::CL_DEVICE_MAX_COMPUTE_UNITS,
        MaxConstantArgs = cl::CL_DEVICE_MAX_CONSTANT_ARGS,
        MaxReadImageArgs = cl::CL_DEVICE_MAX_READ_IMAGE_ARGS,
        MaxSamplers = cl::CL_DEVICE_MAX_SAMPLERS,
        MaxWorkItemDimensions = cl::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
        MaxWriteImageArgs = cl::CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
        MemBaseAddrAlign = cl::CL_DEVICE_MEM_BASE_ADDR_ALIGN,
        MinDataTypeAlignSize = cl::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
        VendorId = cl::CL_DEVICE_VENDOR_ID,
        PreferredVectorWidthChar = cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
        PreferredVectorWidthShort = cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
        PreferredVectorWidthInt = cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
        PreferredVectorWidthLong = cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
        PreferredVectorWidthFloat = cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
        PreferredVectorWidthDouble= cl::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
    }

    impl DeviceInfo for DeviceInfoClUint {
        type Info = cl::cl_uint;
        fn get_device_info(self, device: DeviceId) -> Result<cl::cl_uint> {
            unsafe {
                let mut ret = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, self as cl::cl_device_info,
                    mem::size_of::<cl::cl_uint>() as libc::size_t,
                    &mut ret as *mut _ as *mut _, ptr::null_mut());
                try!(check_status(res));
                Ok(ret)
            }
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct DeviceInfoDeviceType;
    impl DeviceInfo for DeviceInfoDeviceType {
        type Info = DeviceType;
        fn get_device_info(self, device: DeviceId) -> Result<DeviceType> {
            unsafe {
                let mut device_type: cl::cl_device_type = 0;
                let res = cl::ll::clGetDeviceInfo(
                    device.0, cl::CL_DEVICE_TYPE,
                    mem::size_of::<cl::cl_device_type>() as libc::size_t,
                    &mut device_type as *mut _ as *mut _, ptr::null_mut());
                try!(check_status(res));
                match DeviceType::from_bits(device_type) {
                    Some(device_type) => Ok(device_type),
                    None => panic!("Rascal: Got invalid device type {}!", device_type),
                }
            }
        }
    }

    pub fn get_platform_ids() -> Result<Vec<PlatformId>> {
        unsafe {
            let mut num_platforms = 0;
            let res = cl::ll::clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms);
            try!(check_status(res));
            let mut ids: Vec<_> = repeat(0 as *mut _).take(num_platforms as usize).collect();
            let res = cl::ll::clGetPlatformIDs(
                ids.len() as cl::cl_uint, ids.as_mut_ptr(), ptr::null_mut());
            try!(check_status(res));
            Ok(ids.iter().map(|ptr| PlatformId(*ptr)).collect())
        }
    }

    pub fn get_platform_info(platform: PlatformId, info: PlatformInfo) -> Result<String> {
        unsafe {
            let mut info_size = 0;
            let res = cl::ll::clGetPlatformInfo(
                platform.0, info as cl::cl_platform_info, 0, ptr::null_mut(), &mut info_size);
            try!(check_status(res));
            let mut bytes: Vec<_> = repeat(0).take(info_size as usize).collect();
            let res = cl::ll::clGetPlatformInfo(
                platform.0, info as cl::cl_platform_info, bytes.len() as libc::size_t,
                bytes.as_mut_ptr() as *mut _, ptr::null_mut());
            try!(check_status(res));
            Ok(string_from_cstring_buf(bytes))
        }
    }

    pub fn get_device_ids(platform: PlatformId, device_type: DeviceType)
        -> Result<Vec<DeviceId>>
    {
        unsafe {
            let mut num_devices = 0;
            let res = cl::ll::clGetDeviceIDs(
                platform.0, device_type.bits(), 0, ptr::null_mut(), &mut num_devices);
            try!(check_status(res));
            let mut ids: Vec<_> = repeat(0 as *mut _).take(num_devices as usize).collect();
            let res = cl::ll::clGetDeviceIDs(
                platform.0, device_type.bits(), ids.len() as cl::cl_uint, ids.as_mut_ptr(),
                ptr::null_mut());
            try!(check_status(res));
            Ok(ids.iter().map(|ptr| DeviceId(*ptr)).collect())
        }
    }

    pub fn get_device_info<T: DeviceInfo>(device: DeviceId, info: T) -> Result<T::Info> {
        info.get_device_info(device)
    }

    extern "C" fn dummy_handler(errinfo: *const libc::c_char, private_info: *const libc::c_void,
        cb: libc::size_t, user_data: *mut libc::c_void)
    {
        unsafe {
            let _ = private_info;
            let _ = cb;
            let _ = user_data;
            let errinfo = CStr::from_ptr(errinfo).to_str();
            match errinfo {
                Ok(errinfo) => panic!("Rascal: Got error from context! (Error: {})", errinfo),
                Err(err) => panic!(
                    "Rascal: Got error from context! (Error string is invalid: {:?})", err),
            }
        }
    }

    pub fn create_context(platform: PlatformId, devices: &[DeviceId]) -> Result<Context> {
        unsafe {
            let mut err = 0;
            // disgusting and cheaty.
            let props = [
                cl::CL_CONTEXT_PLATFORM as usize as cl::cl_context_properties,
                platform.0 as usize as cl::cl_context_properties,
                0 as cl::cl_context_properties];
            let ids: Vec<_> = devices.iter().map(|d| d.0).collect();
            let context = cl::ll::clCreateContext(
                props.as_ptr(), ids.len() as cl::cl_uint, ids[..].as_ptr(),
                dummy_handler, ptr::null_mut(),
                &mut err as *mut _);
            try!(check_status(err));
            Ok(Context(context))
        }
    }
}

pub mod hl {
    use super::ll;
    use super::Result;

    #[derive(Debug, Copy, Clone)]
    pub struct Platform(ll::PlatformId);

    #[derive(Debug, Copy, Clone)]
    pub struct Device(ll::DeviceId);

    #[derive(Debug)]
    pub struct Context(ll::Context);

    pub fn get_platforms() -> Vec<Platform> {
        ll::get_platform_ids().unwrap().into_iter().map(Platform).collect()
    }

    impl Platform {
        pub fn get_devices(&self) -> Vec<Device> {
            ll::get_device_ids(self.0, ll::device_type::ALL).unwrap()
                .into_iter().map(Device).collect()
        }

        pub fn create_context(&self, devices: &[Device]) -> Result<Context> {
            let devices: Vec<_> = devices.iter().map(|d| d.0).collect();
            ll::create_context(self.0, &devices[..]).map(Context)
        }

        pub fn name(&self) -> String {
            ll::get_platform_info(self.0, ll::PlatformInfo::Name).unwrap()
        }

        pub fn version(&self) -> String {
            ll::get_platform_info(self.0, ll::PlatformInfo::Version).unwrap()
        }

        pub fn profile(&self) -> String {
            ll::get_platform_info(self.0, ll::PlatformInfo::Profile).unwrap()
        }

        pub fn vendor(&self) -> String {
            ll::get_platform_info(self.0, ll::PlatformInfo::Vendor).unwrap()
        }

        pub fn extensions(&self) -> String {
            ll::get_platform_info(self.0, ll::PlatformInfo::Extensions).unwrap()
        }
    }

    impl Device {
        pub fn name(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::Name).unwrap()
        }

        pub fn profile(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::Profile).unwrap()
        }

        pub fn vendor(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::Vendor).unwrap()
        }

        pub fn device_version(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::DeviceVersion).unwrap()
        }

        pub fn driver_version(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::DriverVersion).unwrap()
        }

        pub fn extensions(&self) -> String {
            ll::get_device_info(self.0, ll::DeviceInfoString::Extensions).unwrap()
        }

        pub fn device_type(&self) -> ll::DeviceType {
            ll::get_device_info(self.0, ll::DeviceInfoDeviceType).unwrap()
        }

        pub fn num_compute_units(&self) -> usize {
            ll::get_device_info(self.0, ll::DeviceInfoClUint::MaxComputeUnits).unwrap() as usize
        }
    }
}
