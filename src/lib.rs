#[macro_use]
extern crate bitflags;
extern crate opencl;
extern crate num;
extern crate libc;

pub mod ll;
pub mod hl;

pub type Result<A> = ::std::result::Result<A, opencl::cl::CLStatus>;
