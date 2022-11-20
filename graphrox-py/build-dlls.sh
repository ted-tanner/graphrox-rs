cd ..

# Architectures
############################
# aarch64-apple-darwin
# aarch64-pc-windows-msvc (Windows only)
# aarch64-unknown-linux-gnu
# x86_64-apple-darwin
# x86_64-pc-windows-gnu
# x86_64-unknown-linux-gnu
# x86_64-unknown-freebsd
# x86_64-unknown-netbsd

cargo test --release &&
    cargo build --release --target aarch64-apple-darwin &&
    cargo build --release --target aarch64-unknown-linux-gnu &&
    cargo build --release --target x86_64-apple-darwin &&
    cargo build --release --target x86_64-unknown-linux-gnu
    # cargo build --release --target x86_64-pc-windows-gnu &&
    # cargo build --release --target x86_64-unknown-freebsd &&
    # cargo build --release --target x86_64-unknown-netbsd

cd graphrox-py
