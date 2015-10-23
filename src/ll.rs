use opencl::cl;
use libc;
use std::ffi::CStr;
use std::ptr;
use std::mem;
use std::iter::repeat;
use num;
use super::Result;

fn check_status(status_int: cl::cl_int) -> Result<()> {
    let status = num::FromPrimitive::from_i32(status_int);
    match status {
        Some(cl::CLStatus::CL_SUCCESS) => Ok(()),
        Some(other) => Err(other),
        None => panic!("Rascal: Tried to check invalid opencl status! (Value was {})", status_int)
    }
}

pub use self::device_type::DeviceType;

macro_rules! newtype_to_from_raw {
    ($($(#[$Meta:meta])* pub struct $Name:ident($Type:ty));*;) => {
        $(
            $(#[$Meta])*
            pub struct $Name($Type);

            #[allow(dead_code)]
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

    #[derive(Debug)]
    pub struct Context(cl::cl_context);

    #[derive(Debug)]
    pub struct CommandQueue(cl::cl_command_queue);

    #[derive(Debug)]
    pub struct Mem(cl::cl_mem);

    #[derive(Debug)]
    pub struct Program(cl::cl_program);

    #[derive(Debug)]
    pub struct Kernel(cl::cl_kernel);

    #[derive(Debug)]
    pub struct Event(cl::cl_event);

    #[derive(Debug)]
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
        flags MemFlags: cl::cl_mem_flags {
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
        flags MapFlags: cl::cl_map_flags {
            const READ = cl::CL_MAP_READ,
            const WRITE = cl::CL_MAP_WRITE,
        }
    }
}

pub mod device_type {
    use opencl::cl;
    bitflags! {
        flags DeviceType: cl::cl_device_type {
            const DEFAULT = cl::CL_DEVICE_TYPE_DEFAULT,
            const GPU = cl::CL_DEVICE_TYPE_GPU,
            const CPU = cl::CL_DEVICE_TYPE_CPU,
            const ACCELERATOR = cl::CL_DEVICE_TYPE_ACCELERATOR,
            const ALL = cl::CL_DEVICE_TYPE_ALL,
        }
    }
}

pub mod queue_properties {
    use opencl::cl;
    bitflags! {
        flags QueueProperties: cl::cl_command_queue_properties {
            const OUT_OF_ORDER_EXEC_MODE_ENABLE = cl::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            const PROFILING_ENABLE = cl::CL_QUEUE_PROFILING_ENABLE,
        }
    }
}

// TODO, although none of these are very useful
pub trait ContextInfo {
    type Info;
    fn get_context_info(self, context: &Context) -> Result<Self::Info>;
}

// TODO
pub trait CommandQueueInfo {
    type Info;
    fn get_command_queue_info(self, queue: &CommandQueue) -> Result<Self::Info>;
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

extern "C" fn dummy_context_handler(errinfo: *const libc::c_char,
    private_info: *const libc::c_void, cb: libc::size_t, user_data: *mut libc::c_void)
{
    unsafe {
        let _ = private_info;
        let _ = cb;
        let _ = user_data;
        let errinfo = CStr::from_ptr(errinfo).to_str();
        match errinfo {
            Ok(errinfo) => panic!("Rascal: Got error from OpenCL context! (Error: {})", errinfo),
            Err(err) => panic!(
                "Rascal: Got error from OpenCL context! (Error string is invalid: {:?})", err),
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
            dummy_context_handler, ptr::null_mut(),
            &mut err as *mut _);
        try!(check_status(err));
        Ok(Context(context))
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MemProt {
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

impl MemProt {
    pub fn to_mem_flags(self) -> mem_flags::MemFlags {
        match self {
            MemProt::ReadWrite => mem_flags::READ_WRITE,
            MemProt::ReadOnly => mem_flags::READ_ONLY,
            MemProt::WriteOnly => mem_flags::WRITE_ONLY,
        }
    }
}

// TODO handle the other mem creation cases
/// Creates an entirely device-backed buffer.
pub fn create_mem_device_buffer(context: &Context, permissions: MemProt, size: usize)
    -> Result<Mem>
{
    unsafe {
        let mut err = 0;
        let mem = cl::ll::clCreateBuffer(
            context.0, permissions.to_mem_flags().bits(), size as libc::size_t, ptr::null_mut(),
            &mut err);
        try!(check_status(err));
        Ok(Mem(mem))
    }
}

impl Mem {
    pub fn try_clone(&self) -> Result<Mem> {
        unsafe {
            try!(check_status(cl::ll::clRetainMemObject(self.0)));
            Ok(Mem(self.0))
        }
    }
}

impl Clone for Mem {
    fn clone(&self) -> Mem {
        match self.try_clone() {
            Ok(mem) => mem,
            Err(err) => panic!(
                "Rascal: Failed to increment OpenCL mem object refcount! (Error: {:?})", err),
        }
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        unsafe {
            match check_status(cl::ll::clReleaseMemObject(self.0)) {
                Ok(()) => { }
                Err(err) => panic!(
                    "Rascal: Failed to decrement OpenCL mem object refcount! (Error: {:?})", err)
            }
        }
    }
}

pub fn create_command_queue(context: &Context, device: DeviceId,
    properties: queue_properties::QueueProperties)
    -> Result<CommandQueue>
{
    unsafe {
        let mut err = 0;
        let queue = cl::ll::clCreateCommandQueue(
            context.0, device.0, properties.bits(), &mut err);
        try!(check_status(err));
        Ok(CommandQueue(queue))
    }
}

impl CommandQueue {
    pub fn try_clone(&self) -> Result<CommandQueue> {
        unsafe {
            try!(check_status(cl::ll::clRetainCommandQueue(self.0)));
            Ok(CommandQueue(self.0))
        }
    }
}

impl Clone for CommandQueue {
    fn clone(&self) -> CommandQueue {
        match self.try_clone() {
            Ok(queue) => queue,
            Err(err) => panic!(
                "Rascal: Failed to increment OpenCL command queue refcount! (Error: {:?})", err),
        }
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe {
            match check_status(cl::ll::clReleaseCommandQueue(self.0)) {
                Ok(()) => { }
                Err(err) => panic!(
                    "Rascal: Failed to decrement OpenCL command queue refcount! (Error: {:?})",
                    err)
            }
        }
    }
}

impl Context {
    pub fn try_clone(&self) -> Result<Context> {
        unsafe {
            try!(check_status(cl::ll::clRetainContext(self.0)));
            Ok(Context(self.0))
        }
    }
}

impl Clone for Context {
    fn clone(&self) -> Context {
        match self.try_clone() {
            Ok(context) => context,
            Err(err) => panic!(
                "Rascal: Failed to increment OpenCL context refcount! (Error: {:?})", err),
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            match check_status(cl::ll::clReleaseContext(self.0)) {
                Ok(()) => { }
                Err(err) => panic!(
                    "Rascal: Failed to decrement OpenCL context refcount! (Error: {:?})", err)
            }
        }
    }
}
