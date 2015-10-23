use super::ll;
use super::Result;

#[derive(Debug, Copy, Clone)]
pub struct Platform(ll::PlatformId);

#[derive(Debug, Copy, Clone)]
pub struct Device(ll::DeviceId);

#[derive(Debug, Clone)]
pub struct Context(ll::Context);

pub fn get_platforms() -> Vec<Platform> {
    ll::get_platform_ids().unwrap().into_iter().map(Platform).collect()
}

impl Context {
    
}

impl Platform {
    pub fn get_devices(&self) -> Vec<Device> {
        ll::get_device_ids(self.0, ll::device_type::ALL).unwrap()
            .into_iter().map(Device).collect()
    }

    pub fn create_context(&self, devices: &[Device]) -> Result<Context> {
        // yes, this double-buffers (and so does ll::create_context).
        // this is because transmute is the devil, and I will avoid it for
        // as long as I can.
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
