extern crate rascal;

fn main() {
    use rascal::types::hl;
    let platform = hl::get_platforms()[0];
    let device = platform.get_devices()[0];
    let context = platform.create_context(&[device]).unwrap();
}
