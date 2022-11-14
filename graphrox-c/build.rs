use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    const GPHRX_HEADER_FILE_NAME: &'static str = "gphrx.h";

    let crate_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR environment variable should be set"),
    );

    let target_dir = match env::var("CARGO_TARGET_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            // Cargo doesn't provide a way to get the target directory, so we have to hackishly
            // guess based on where the OUT_DIR is. If we guess wrong, we fail. This script
            // just copies a file anyway, so it is not needed necessarily. If this is failing,
            // just turn off the build script in Cargo.toml and copy the header file manually.
            let mut dir = PathBuf::from(
                env::var("OUT_DIR").expect("OUT_DIR environment variable should be set"),
            );

            for _ in 0..3 {
                dir.pop();
            }

            let mut top_target_dir = dir.clone();
            top_target_dir.pop();

            match top_target_dir.file_name() {
                Some(dir_name) if dir_name == "target" => (),
                Some(_) | None => panic!("Build script was unable to find target directory"),
            }

            dir
        }
    };

    let header_file_path = crate_dir.join(GPHRX_HEADER_FILE_NAME);
    let target_file_path = target_dir.join(GPHRX_HEADER_FILE_NAME);

    fs::copy(header_file_path, target_file_path).unwrap();
}
