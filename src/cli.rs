//! Zero-dependency <5 ms cold-start CLI driver for `cachepin`.

use std::io::{self, Read};
use std::path::PathBuf;

use crate::audit::{audit_jsonl_file, audit_multiple_paths, diff_two_requests};
use crate::bench::run_full_benchmark_suite;
use crate::canonical::CompilerConfig;
use crate::mcp::run_mcp_stdio;
use crate::pricing::list_supported_models;
use crate::proxy::{run_proxy_blocking, ProxyConfig};
use crate::router::RoutingPolicy;
use crate::simd::simd_backend_name;
use crate::types::{CanonicalMode, ProviderFormat};
use crate::{CachePinEngine, VERSION};

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        "-V" | "--version" | "version" => {
            println!("cachepin {} (simd: {})", VERSION, simd_backend_name());
            Ok(())
        }
        "proxy" => cmd_proxy(&args[2..]),
        "audit" => cmd_audit(&args[2..]),
        "diff" => cmd_diff(&args[2..]),
        "optimize" => cmd_optimize(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "mcp" => cmd_mcp(&args[2..]),
        "models" => cmd_models(),
        other => Err(format!(
            "unknown subcommand '{}'. Run `cachepin --help` for usage.",
            other
        )),
    }
}

fn print_help() {
    println!(
        r#"cachepin {} — Pin your prefix. Stop burning 20x on broken multi-agent LLM prompt caches.

USAGE:
  cachepin <COMMAND> [OPTIONS]

COMMANDS:
  proxy      Start the low-latency HTTP/SSE caching & swarm-coalescing reverse proxy
  audit      Scan JSONL traces, Claude Code (~/.claude/projects), or Codex (~/.codex/sessions)
  diff       SIMD byte-level prefix divergence & counterfactual recovery between two JSON payloads
  optimize   Canonicalize and inject exact-DP breakpoints into a single JSON request (stdin/file)
  bench      Run reproducible [LIVE-CPU] and [SIM-TRACE] benchmarks
  mcp        Start native stdio Model Context Protocol (MCP) server for coding agents
  models     List built-in 2026 provider pricing & cache geometry profiles
  version    Print version and active SIMD backend ({})

EXAMPLES:
  cachepin audit --auto
  cachepin audit --claude-code
  cachepin audit --codex
  cachepin diff agent_a.json agent_b.json
  cachepin proxy --upstream https://api.anthropic.com --port 47820
  cachepin proxy --backends http://gpu1:8000,http://gpu2:8000 --kv-tokens 131072
  cat request.json | cachepin optimize --mode swarm
  cachepin bench --json
  cachepin mcp"#,
        VERSION,
        simd_backend_name()
    );
}

fn cmd_proxy(args: &[String]) -> Result<(), String> {
    let mut cfg = ProxyConfig::default();
    let mut port: u16 = 47820;
    let mut host = "127.0.0.1".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or("--port requires a value")?
                    .parse()
                    .map_err(|_| "invalid --port")?;
            }
            "--host" => {
                i += 1;
                host = args.get(i).ok_or("--host requires a value")?.clone();
            }
            "--upstream" | "-u" => {
                i += 1;
                let u = args.get(i).ok_or("--upstream requires a value")?.clone();
                cfg.upstreams = vec![u];
            }
            "--backends" | "-b" => {
                i += 1;
                let list = args.get(i).ok_or("--backends requires a comma-separated list")?;
                cfg.upstreams = list
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--mode" | "-m" => {
                i += 1;
                cfg.mode = CanonicalMode::from_str_loose(args.get(i).ok_or("--mode requires a value")?);
            }
            "--provider" => {
                i += 1;
                cfg.provider_hint = Some(ProviderFormat::from_str_loose(
                    args.get(i).ok_or("--provider requires a value")?,
                ));
            }
            "--no-coalesce" => cfg.enable_coalescing = false,
            "--no-breakpoints" => cfg.inject_breakpoints = false,
            "--coalesce-wait-ms" => {
                i += 1;
                cfg.coalesce_max_wait_ms = args
                    .get(i)
                    .ok_or("--coalesce-wait-ms requires a value")?
                    .parse()
                    .map_err(|_| "invalid --coalesce-wait-ms")?;
            }
            "--kv-tokens" => {
                i += 1;
                cfg.kv_capacity_tokens = args
                    .get(i)
                    .ok_or("--kv-tokens requires a value")?
                    .parse()
                    .map_err(|_| "invalid --kv-tokens")?;
            }
            "--policy" => {
                i += 1;
                cfg.routing_policy = match args.get(i).ok_or("--policy requires a value")?.as_str() {
                    "rr" | "round-robin" => RoutingPolicy::RoundRobin,
                    "least-loaded" | "ll" => RoutingPolicy::LeastLoaded,
                    "hash" | "consistent-hash" => RoutingPolicy::ConsistentHash,
                    "single-home" => RoutingPolicy::SingleHomePrefixDivert,
                    _ => RoutingPolicy::CachePinAdaptiveReplication,
                };
            }
            "-h" | "--help" => {
                println!("Usage: cachepin proxy [--host 127.0.0.1] [--port 47820] [--upstream URL | --backends URL1,URL2] [--mode observe|json|hoist|swarm] [--no-coalesce] [--no-breakpoints] [--kv-tokens 131072]");
                return Ok(());
            }
            other => return Err(format!("unknown proxy flag '{}'", other)),
        }
        i += 1;
    }
    cfg.listen_addr = format!("{}:{}", host, port);
    run_proxy_blocking(cfg)
}

fn resolve_home_dir() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| "cannot resolve HOME directory".to_string())
}

fn cmd_audit(args: &[String]) -> Result<(), String> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut mode = CanonicalMode::SwarmAlign;
    let mut provider: Option<ProviderFormat> = None;
    let mut as_json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" | "-m" => {
                i += 1;
                mode = CanonicalMode::from_str_loose(args.get(i).ok_or("--mode requires value")?);
            }
            "--provider" => {
                i += 1;
                provider = Some(ProviderFormat::from_str_loose(
                    args.get(i).ok_or("--provider requires value")?,
                ));
            }
            "--claude-code" => {
                let home = resolve_home_dir()?;
                paths.push(home.join(".claude").join("projects"));
            }
            "--codex" => {
                let home = resolve_home_dir()?;
                paths.push(home.join(".codex").join("sessions"));
            }
            "--auto" => {
                let home = resolve_home_dir()?;
                paths.push(home.join(".claude").join("projects"));
                paths.push(home.join(".codex").join("sessions"));
            }
            "--json" => as_json = true,
            "-h" | "--help" => {
                println!("Usage: cachepin audit (<trace.jsonl|dir> | --claude-code | --codex | --auto) [--mode json|hoist|swarm] [--provider anthropic|openai|vllm] [--json]");
                return Ok(());
            }
            p if !p.starts_with('-') => paths.push(PathBuf::from(p)),
            other => return Err(format!("unknown audit flag '{}'", other)),
        }
        i += 1;
    }
    if paths.is_empty() {
        return Err(
            "missing <trace.jsonl|dir> path or --claude-code / --codex / --auto. Run `cachepin audit --help`."
                .to_string(),
        );
    }
    let report = if paths.len() == 1 {
        audit_jsonl_file(&paths[0], mode, provider)?
    } else {
        audit_multiple_paths(&paths, mode, provider)?
    };
    if as_json {
        println!("{}", report.to_json_string());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn cmd_diff(args: &[String]) -> Result<(), String> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut mode = CanonicalMode::SwarmAlign;
    let mut provider: Option<ProviderFormat> = None;
    let mut as_json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" | "-m" => {
                i += 1;
                mode = CanonicalMode::from_str_loose(args.get(i).ok_or("--mode requires value")?);
            }
            "--provider" => {
                i += 1;
                provider = Some(ProviderFormat::from_str_loose(
                    args.get(i).ok_or("--provider requires value")?,
                ));
            }
            "--json" => as_json = true,
            "-h" | "--help" => {
                println!("Usage: cachepin diff <req_a.json> <req_b.json> [--mode json|hoist|swarm] [--provider anthropic|openai|vllm] [--json]");
                return Ok(());
            }
            p if !p.starts_with('-') => files.push(PathBuf::from(p)),
            other => return Err(format!("unknown diff flag '{}'", other)),
        }
        i += 1;
    }
    if files.len() != 2 {
        return Err("expected exactly 2 JSON files: `cachepin diff <req_a.json> <req_b.json>`".to_string());
    }
    let a = std::fs::read(&files[0])
        .map_err(|e| format!("failed to read {}: {}", files[0].display(), e))?;
    let b = std::fs::read(&files[1])
        .map_err(|e| format!("failed to read {}: {}", files[1].display(), e))?;
    let report = diff_two_requests(&a, &b, mode, provider)?;
    if as_json {
        println!("{}", report.to_json_string());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn cmd_optimize(args: &[String]) -> Result<(), String> {
    let mut mode = CanonicalMode::SwarmAlign;
    let mut provider: Option<ProviderFormat> = None;
    let mut inject = true;
    let mut input_file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" | "-m" => {
                i += 1;
                mode = CanonicalMode::from_str_loose(args.get(i).ok_or("--mode requires value")?);
            }
            "--provider" => {
                i += 1;
                provider = Some(ProviderFormat::from_str_loose(
                    args.get(i).ok_or("--provider requires value")?,
                ));
            }
            "--no-breakpoints" => inject = false,
            "-h" | "--help" => {
                println!("Usage: cachepin optimize [request.json] [--mode json|hoist|swarm] [--no-breakpoints]");
                return Ok(());
            }
            p if !p.starts_with('-') => input_file = Some(PathBuf::from(p)),
            other => return Err(format!("unknown optimize flag '{}'", other)),
        }
        i += 1;
    }
    let raw = match input_file {
        Some(p) => std::fs::read(&p).map_err(|e| format!("failed to read {}: {}", p.display(), e))?,
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| e.to_string())?;
            buf
        }
    };
    let mut engine = CachePinEngine::new(mode, 1024);
    let out = engine.process_request("cli", &raw, provider, 1_700_000_000_000, inject)?;
    println!(
        "{}",
        String::from_utf8_lossy(&out.optimized_json)
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let as_json = args.iter().any(|a| a == "--json");
    let res = run_full_benchmark_suite()?;
    if as_json {
        println!("{}", res.to_json_string());
    } else {
        print!("{}", res.format_markdown());
    }
    Ok(())
}

fn cmd_mcp(args: &[String]) -> Result<(), String> {
    let mut cfg = CompilerConfig::default();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--mode" {
            i += 1;
            cfg.mode = CanonicalMode::from_str_loose(args.get(i).ok_or("--mode requires value")?);
        }
        i += 1;
    }
    run_mcp_stdio(cfg)
}

fn cmd_models() -> Result<(), String> {
    println!(
        "{:<20} {:<16} {:>8} {:>8} {:>8} {:>8} {:>7} {:>4}",
        "MODEL PREFIX", "PROVIDER", "IN $/M", "5M $/M", "1H $/M", "HIT $/M", "MIN TOK", "BPS"
    );
    println!("{}", "-".repeat(86));
    for m in list_supported_models() {
        println!(
            "{:<20} {:<16} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7} {:>4}",
            m.model_prefix,
            m.provider.as_str(),
            m.input_per_mtok,
            m.write_5m_per_mtok,
            m.write_1h_per_mtok,
            m.read_per_mtok,
            m.min_cache_tokens,
            m.max_breakpoints
        );
    }
    Ok(())
}
