"""cachepin — Zero-dependency Python SDK backed by native Rust/AVX2 `libcachepin.so`.

Quick start:
    from cachepin import CachePin
    cp = CachePin(mode="swarm")
    out = cp.compile_and_plan(request_dict, tenant="team-alpha", provider="anthropic")
    optimized_payload = out["canonical_json"]
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

__version__ = "0.1.0"

_MODE_MAP = {
    "observe": 0,
    "observe-only": 0,
    "json": 1,
    "json-canonical": 1,
    "strict": 1,
    "hoist": 2,
    "hoist-volatile": 2,
    "swarm": 3,
    "swarm-align": 3,
}

_PROVIDER_MAP = {
    "auto": 0,
    "anthropic": 1,
    "claude": 1,
    "openai": 2,
    "openai-chat": 2,
    "openai-responses": 3,
    "responses": 3,
    "vllm": 4,
    "sglang": 4,
    "generic": 0,
}


def _find_library() -> Path:
    env_path = os.environ.get("CACHEPIN_LIB_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    ext = ".dylib" if sys.platform == "darwin" else ".dll" if os.name == "nt" else ".so"
    lib_name = f"cachepin{ext}" if os.name == "nt" else f"libcachepin{ext}"
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    candidates = [
        pkg_dir / lib_name,
        repo_root / "target" / "release" / lib_name,
        repo_root / "target" / "debug" / lib_name,
        Path.cwd() / "target" / "release" / lib_name,
        Path.cwd() / "target" / "debug" / lib_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Could not find {lib_name}. Build with `cargo build --release` or set CACHEPIN_LIB_PATH."
    )


class _NativeBindings:
    def __init__(self, lib_path: Optional[Union[str, Path]] = None) -> None:
        path = Path(lib_path) if lib_path else _find_library()
        self.lib_path = path
        lib = ctypes.CDLL(str(path))
        self.lib = lib

        lib.cachepin_engine_new.argtypes = [ctypes.c_uint32, ctypes.c_size_t]
        lib.cachepin_engine_new.restype = ctypes.c_void_p

        lib.cachepin_engine_free.argtypes = [ctypes.c_void_p]
        lib.cachepin_engine_free.restype = None

        lib.cachepin_compile_and_plan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.cachepin_compile_and_plan.restype = ctypes.c_void_p

        lib.cachepin_lcp_len.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.cachepin_lcp_len.restype = ctypes.c_size_t

        lib.cachepin_string_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.cachepin_string_free.restype = None

        lib.cachepin_version.argtypes = []
        lib.cachepin_version.restype = ctypes.c_char_p


_NATIVE_CACHE: Optional[_NativeBindings] = None


def _get_native(lib_path: Optional[Union[str, Path]] = None) -> _NativeBindings:
    global _NATIVE_CACHE
    if lib_path is not None:
        return _NativeBindings(lib_path)
    if _NATIVE_CACHE is None:
        _NATIVE_CACHE = _NativeBindings()
    return _NATIVE_CACHE


def lcp_len(a: bytes, b: bytes) -> int:
    """Compute Longest Common Prefix length using AVX2 SIMD via libcachepin."""
    native = _get_native()
    return int(native.lib.cachepin_lcp_len(a, len(a), b, len(b)))


def native_version() -> str:
    """Return compiled native engine version string."""
    native = _get_native()
    raw = native.lib.cachepin_version()
    return raw.decode("utf-8") if raw else __version__


class CachePin:
    """Stateful tenant-isolated prompt-cache compiler & exact DP breakpoint planner."""

    def __init__(
        self,
        mode: str = "swarm",
        max_radix_nodes: int = 65_536,
        lib_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._native = _get_native(lib_path)
        mode_code = _MODE_MAP.get(mode.lower(), 3)
        handle = self._native.lib.cachepin_engine_new(mode_code, int(max_radix_nodes))
        if not handle:
            raise RuntimeError("cachepin_engine_new returned NULL")
        self._handle = handle

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._native.lib.cachepin_engine_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "CachePin":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def compile_and_plan(
        self,
        payload: Union[bytes, str, Dict[str, Any]],
        tenant: str = "default",
        provider: str = "auto",
    ) -> Dict[str, Any]:
        """Canonicalize a request, diagnose cache misses, and solve optimal breakpoints."""
        if not self._handle:
            raise RuntimeError("CachePin engine is closed")
        if isinstance(payload, dict):
            raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        elif isinstance(payload, str):
            raw_bytes = payload.encode("utf-8")
        else:
            raw_bytes = bytes(payload)

        tenant_bytes = tenant.encode("utf-8")
        prov_code = _PROVIDER_MAP.get(provider.lower(), 0)
        out_len = ctypes.c_size_t(0)
        ptr = self._native.lib.cachepin_compile_and_plan(
            self._handle,
            tenant_bytes,
            len(tenant_bytes),
            raw_bytes,
            len(raw_bytes),
            prov_code,
            ctypes.byref(out_len),
        )
        if not ptr:
            raise RuntimeError("cachepin_compile_and_plan returned NULL")
        try:
            data = ctypes.string_at(ptr, out_len.value)
        finally:
            self._native.lib.cachepin_string_free(ptr, out_len.value)
        return json.loads(data.decode("utf-8"))

    def optimize_dict(
        self,
        payload: Dict[str, Any],
        tenant: str = "default",
        provider: str = "auto",
    ) -> Dict[str, Any]:
        """Drop-in helper returning the optimized request dict ready for Anthropic/OpenAI SDKs."""
        res = self.compile_and_plan(payload, tenant=tenant, provider=provider)
        if "error" in res:
            raise ValueError(res["error"])
        opt = res.get("optimized_json") or res.get("canonical_json")
        if isinstance(opt, str):
            return json.loads(opt)
        if isinstance(opt, dict):
            return opt
        return payload


__all__ = ["CachePin", "lcp_len", "native_version", "__version__"]
