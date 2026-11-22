fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(err) = cachepin::cli::run_cli(&args) {
        eprintln!("cachepin error: {}", err);
        std::process::exit(1);
    }
}
