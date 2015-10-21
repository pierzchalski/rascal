extern crate rascal;
extern crate opencl;

fn main() {
    {
        use opencl::hl;
        println!("Using opencl...");
        for platform in hl::get_platforms().iter() {
            println!("Platform: {}", platform.name());
            println!("Platform Version: {}", platform.version());
            println!("Vendor:   {}", platform.vendor());
            println!("Profile:  {}", platform.profile());
            println!("Available extensions: {}", platform.extensions());
            println!("Available devices:");
            for device in platform.get_devices().iter() {
                println!("   Name: {}", device.name());
                println!("   Type: {}", device.device_type());
                println!("   Profile: {}", device.profile());
                println!("   Compute Units: {}", device.compute_units());
            }
        }
    }
    println!("");
    {
        println!("Using rascal...");
        for platform in rascal::types::hl::get_platforms().iter() {
            println!("Platform: {}", platform.name());
            println!("Platform Version: {}", platform.version());
            println!("Vendor:   {}", platform.vendor());
            println!("Profile:  {}", platform.profile());
            println!("Available extensions: {}", platform.extensions());
            println!("Available devices:");
            for device in platform.get_devices().iter() {
                println!("   Name: {}", device.name());
                println!("   Type: {:?}", device.device_type());
                println!("   Profile: {}", device.profile());
                //println!("   Compute Units: {}", device.compute_units());
            }
        }
    }
}
