class Cachepin < Formula
  desc "Zero-dependency Rust + SIMD lossless multi-agent LLM prompt/KV-cache compiler & proxy"
  homepage "https://pgg898200-ship-it.github.io/cachepin/"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/pgg898200-ship-it/cachepin.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    output = shell_output("#{bin}/cachepin --version")
    assert_match "cachepin", output
  end
end
