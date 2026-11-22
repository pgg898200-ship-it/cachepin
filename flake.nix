{
  description = "cachepin — Zero-dependency Rust + SIMD multi-agent LLM prompt-cache compiler, exact DP solver, coalescing proxy & Shadow-KV router";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      packages = forAllSystems (pkgs: rec {
        cachepin = pkgs.rustPlatform.buildRustPackage {
          pname = "cachepin";
          version = "0.2.0";
          src = self;
          cargoLock.lockFile = ./Cargo.lock;
          meta = with pkgs.lib; {
            description = "Lossless multi-agent LLM prompt/KV-cache compiler, exact mixed-TTL DP solver, auditor, and Shadow-KV router";
            homepage = "https://pgg898200-ship-it.github.io/cachepin/";
            license = with licenses; [ mit asl20 ];
            mainProgram = "cachepin";
          };
        };
        default = cachepin;
      });

      apps = forAllSystems (pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.cachepin}/bin/cachepin";
        };
      });
    };
}
