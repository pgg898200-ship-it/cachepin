//! Wait-free fast-path, single-flight thundering-herd coalescer triggered at TTFT.
//!
//! When a swarm of concurrent agents sends requests sharing the same cold prompt
//! prefix `(tenant_key, model_hash, prefix_hash)`, naive forwarding triggers `N`
//! redundant cache-write surcharges simultaneously. `SingleFlightCoalescer` elects
//! one `Leader` to warm the upstream KV cache while `N - 1` followers park on a
//! per-key `Condvar` (zero busy spinning). As soon as the leader observes upstream
//! response headers / first SSE chunk (TTFT), it calls
//! [`LeaderGuard::mark_warm_and_release`], waking all followers simultaneously and
//! placing the key on the ~50 ns `BypassWarm` fast path for `warm_ttl_ms`.
//!
//! RAII `Drop` guarantees fail-open semantics: if the leader fails, errors, or
//! panics before warming the cache, all followers are woken immediately with
//! `leader_warmed = false`.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Upper bound on cached warm-prefix timestamps before stale eviction runs.
const MAX_WARM_ENTRIES: usize = 65_536;

/// Lightweight randomized `u64` hasher using SplitMix64 mixing so keyed
/// `flight_key` lookups avoid SipHash overhead while remaining collision-safe.
#[derive(Clone, Copy)]
struct FastU64BuildHasher {
    seed: u64,
}

impl FastU64BuildHasher {
    fn new() -> Self {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(0x9e37_79b9_7f4a_7c15);
        Self { seed: h.finish() }
    }
}

struct FastU64Hasher {
    state: u64,
}

impl Hasher for FastU64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state = self
                .state
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(b as u64);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state ^= i;
    }
}

impl BuildHasher for FastU64BuildHasher {
    type Hasher = FastU64Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        FastU64Hasher { state: self.seed }
    }
}

/// Snapshot of single-flight coalescer telemetry counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoalescerStats {
    /// Total leader elections granted (cold prefixes forwarded to warm cache).
    pub leaders: u64,
    /// Total follower requests that joined an in-flight leader.
    pub followers_coalesced: u64,
    /// Followers woken after the leader confirmed upstream cache warming (TTFT).
    pub followers_warmed: u64,
    /// Followers that hit `max_wait_ms` and proceeded fail-open.
    pub followers_timed_out: u64,
    /// Requests served immediately via the warm-key fast path without blocking.
    pub warm_bypasses: u64,
    /// Cumulative milliseconds followers spent waiting for leaders.
    pub total_wait_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    InFlight,
    Finished { warmed: bool },
}

struct FlightSlot {
    state: Mutex<SlotState>,
    cvar: Condvar,
}

impl FlightSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::InFlight),
            cvar: Condvar::new(),
        }
    }
}

struct FlightTable {
    warm_until: HashMap<u64, u64, FastU64BuildHasher>,
    inflight: HashMap<u64, Arc<FlightSlot>, FastU64BuildHasher>,
}

impl FlightTable {
    fn new() -> Self {
        let hasher = FastU64BuildHasher::new();
        Self {
            warm_until: HashMap::with_hasher(hasher),
            inflight: HashMap::with_hasher(hasher),
        }
    }

    fn prune_warm_if_needed(&mut self, now_ms: u64) {
        if self.warm_until.len() < MAX_WARM_ENTRIES {
            return;
        }
        self.warm_until.retain(|_, &mut until| until > now_ms);
        if self.warm_until.len() < MAX_WARM_ENTRIES {
            return;
        }
        // Still at capacity with active entries: evict the older half by expiry time.
        let mut min_until = u64::MAX;
        let mut max_until = 0u64;
        for &until in self.warm_until.values() {
            if until < min_until {
                min_until = until;
            }
            if until > max_until {
                max_until = until;
            }
        }
        let midpoint = min_until.saturating_add(max_until.saturating_sub(min_until) / 2);
        self.warm_until.retain(|_, &mut until| until > midpoint);
        if self.warm_until.len() >= MAX_WARM_ENTRIES {
            // Degenerate case where all expiry timestamps are identical: clear half deterministically.
            let target = MAX_WARM_ENTRIES / 2;
            let keys_to_drop: Vec<u64> = self
                .warm_until
                .keys()
                .take(self.warm_until.len().saturating_sub(target))
                .copied()
                .collect();
            for k in keys_to_drop {
                self.warm_until.remove(&k);
            }
        }
    }
}

struct CoalescerInner {
    max_wait_ms: u64,
    warm_ttl_ms: u64,
    table: RwLock<FlightTable>,
    leaders: AtomicU64,
    followers_coalesced: AtomicU64,
    followers_warmed: AtomicU64,
    followers_timed_out: AtomicU64,
    warm_bypasses: AtomicU64,
    total_wait_ms: AtomicU64,
}

impl CoalescerInner {
    /// Finishes an in-flight slot. Lock ordering invariant: `self.table` and
    /// `slot.state` are NEVER held simultaneously, eliminating any possibility
    /// of lock-ordering deadlock.
    fn finish_flight(&self, flight_key: u64, slot: &Arc<FlightSlot>, warm_now_ms: Option<u64>) {
        let warmed = warm_now_ms.is_some();

        // Phase 1: Update routing table (mark warm if requested, remove in-flight slot).
        {
            let mut tbl = self.table.write().unwrap_or_else(|e| e.into_inner());
            if let Some(now_ms) = warm_now_ms {
                tbl.prune_warm_if_needed(now_ms);
                let until = now_ms.saturating_add(self.warm_ttl_ms);
                tbl.warm_until.insert(flight_key, until);
            }
            let should_remove = tbl
                .inflight
                .get(&flight_key)
                .map(|existing| Arc::ptr_eq(existing, slot))
                .unwrap_or(false);
            if should_remove {
                tbl.inflight.remove(&flight_key);
            }
        }

        // Phase 2: Transition slot state and wake all parked followers.
        {
            let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            *st = SlotState::Finished { warmed };
            slot.cvar.notify_all();
        }
    }

    fn wait_as_follower(&self, slot: Arc<FlightSlot>) -> FlightRole {
        self.followers_coalesced.fetch_add(1, Ordering::Relaxed);

        let start = Instant::now();
        let max_wait = Duration::from_millis(self.max_wait_ms);
        let deadline = start.checked_add(max_wait);

        let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match *st {
                SlotState::Finished { warmed } => {
                    let waited_ms = start.elapsed().as_millis().min(u64::MAX as u128) as u64;
                    self.total_wait_ms.fetch_add(waited_ms, Ordering::Relaxed);
                    if warmed {
                        self.followers_warmed.fetch_add(1, Ordering::Relaxed);
                    }
                    return FlightRole::FollowerWaited {
                        waited_ms,
                        leader_warmed: warmed,
                    };
                }
                SlotState::InFlight => {
                    let remaining = match deadline {
                        Some(dl) => {
                            let now = Instant::now();
                            if now >= dl {
                                let waited_ms =
                                    start.elapsed().as_millis().min(u64::MAX as u128) as u64;
                                self.total_wait_ms.fetch_add(waited_ms, Ordering::Relaxed);
                                self.followers_timed_out.fetch_add(1, Ordering::Relaxed);
                                return FlightRole::FollowerWaited {
                                    waited_ms,
                                    leader_warmed: false,
                                };
                            }
                            dl.saturating_duration_since(now)
                        }
                        None => Duration::from_secs(3600),
                    };

                    let (guard, timeout_res) = slot
                        .cvar
                        .wait_timeout(st, remaining)
                        .unwrap_or_else(|e| e.into_inner());
                    st = guard;

                    if timeout_res.timed_out() {
                        // Re-check state in case leader signaled right at the timeout boundary.
                        if let SlotState::Finished { warmed } = *st {
                            let waited_ms =
                                start.elapsed().as_millis().min(u64::MAX as u128) as u64;
                            self.total_wait_ms.fetch_add(waited_ms, Ordering::Relaxed);
                            if warmed {
                                self.followers_warmed.fetch_add(1, Ordering::Relaxed);
                            }
                            return FlightRole::FollowerWaited {
                                waited_ms,
                                leader_warmed: warmed,
                            };
                        }
                        let waited_ms = start.elapsed().as_millis().min(u64::MAX as u128) as u64;
                        self.total_wait_ms.fetch_add(waited_ms, Ordering::Relaxed);
                        self.followers_timed_out.fetch_add(1, Ordering::Relaxed);
                        return FlightRole::FollowerWaited {
                            waited_ms,
                            leader_warmed: false,
                        };
                    }
                }
            }
        }
    }
}

/// RAII guard returned to the single elected leader for a cold prefix flight.
///
/// Call [`LeaderGuard::mark_warm_and_release`] immediately when upstream TTFT
/// (response headers or first SSE chunk) confirms the KV prefix is populated.
/// If dropped without calling `mark_warm_and_release`, `Drop` automatically
/// releases the slot and wakes all followers fail-open (`leader_warmed = false`).
pub struct LeaderGuard {
    inner: Arc<CoalescerInner>,
    flight_key: u64,
    slot: Arc<FlightSlot>,
    released: bool,
}

impl LeaderGuard {
    /// Marks `flight_key` warm until `now_ms + warm_ttl_ms`, removes the
    /// in-flight entry, and wakes every parked follower via `notify_all()`.
    pub fn mark_warm_and_release(mut self, now_ms: u64) {
        if !self.released {
            self.released = true;
            self.inner
                .finish_flight(self.flight_key, &self.slot, Some(now_ms));
        }
    }

    /// Explicitly releases the flight without marking the prefix warm
    /// (identical to dropping the guard).
    pub fn release_without_warming(mut self) {
        if !self.released {
            self.released = true;
            self.inner.finish_flight(self.flight_key, &self.slot, None);
        }
    }

    /// Returns the flight key owned by this guard.
    #[inline]
    pub fn flight_key(&self) -> u64 {
        self.flight_key
    }
}

impl Drop for LeaderGuard {
    fn drop(&mut self) {
        if !self.released {
            self.released = true;
            self.inner.finish_flight(self.flight_key, &self.slot, None);
        }
    }
}

impl fmt::Debug for LeaderGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeaderGuard")
            .field("flight_key", &self.flight_key)
            .field("released", &self.released)
            .finish()
    }
}

/// Role assigned by [`SingleFlightCoalescer::acquire`].
#[derive(Debug)]
pub enum FlightRole {
    /// Caller is the elected leader responsible for warming the upstream prefix.
    Leader(LeaderGuard),
    /// Caller coalesced behind an active leader and resumed after wait or release.
    FollowerWaited { waited_ms: u64, leader_warmed: bool },
    /// Prefix is already known warm; caller bypassed coalescing immediately.
    BypassWarm,
}

/// Thread-safe, TTFT-triggered single-flight thundering-herd coalescer.
#[derive(Clone)]
pub struct SingleFlightCoalescer {
    inner: Arc<CoalescerInner>,
}

impl fmt::Debug for SingleFlightCoalescer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleFlightCoalescer")
            .field("max_wait_ms", &self.inner.max_wait_ms)
            .field("warm_ttl_ms", &self.inner.warm_ttl_ms)
            .field("stats", &self.stats())
            .finish()
    }
}

impl SingleFlightCoalescer {
    /// Creates a new coalescer where followers wait at most `max_wait_ms` for
    /// a leader's TTFT signal, and warmed keys stay on the fast bypass path for
    /// `warm_ttl_ms`.
    pub fn new(max_wait_ms: u64, warm_ttl_ms: u64) -> Self {
        Self {
            inner: Arc::new(CoalescerInner {
                max_wait_ms,
                warm_ttl_ms,
                table: RwLock::new(FlightTable::new()),
                leaders: AtomicU64::new(0),
                followers_coalesced: AtomicU64::new(0),
                followers_warmed: AtomicU64::new(0),
                followers_timed_out: AtomicU64::new(0),
                warm_bypasses: AtomicU64::new(0),
                total_wait_ms: AtomicU64::new(0),
            }),
        }
    }

    /// Combines `(tenant_key, model_hash, prefix_hash)` into a single 64-bit
    /// flight key using bijective SplitMix64 avalanche mixing.
    #[inline]
    pub fn compute_flight_key(tenant_key: u64, model_hash: u64, prefix_hash: u64) -> u64 {
        let mut x = tenant_key
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
            ^ model_hash.wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_right(13)
            ^ prefix_hash.wrapping_mul(0x94d0_49bb_1331_11eb);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// Acquires a flight role for `flight_key` at logical/wall timestamp `now_ms`.
    ///
    /// - Warm fast path (`now_ms < warm_until_ms`): shared read lock only,
    ///   returns [`FlightRole::BypassWarm`] immediately.
    /// - Active leader exists: clones the per-key `Arc<FlightSlot>`, releases the
    ///   table lock, and parks on the slot's `Condvar` until TTFT or `max_wait_ms`.
    /// - Cold miss: upgrades to write lock with double-checked locking, inserts a
    ///   new slot, and returns [`FlightRole::Leader`].
    pub fn acquire(&self, flight_key: u64, now_ms: u64) -> FlightRole {
        // Fast path: shared read lock for warm bypass or joining an active flight.
        let existing_slot = {
            let tbl = self.inner.table.read().unwrap_or_else(|e| e.into_inner());
            if let Some(&warm_until) = tbl.warm_until.get(&flight_key) {
                if now_ms < warm_until {
                    self.inner.warm_bypasses.fetch_add(1, Ordering::Relaxed);
                    return FlightRole::BypassWarm;
                }
            }
            tbl.inflight.get(&flight_key).cloned()
        };

        if let Some(slot) = existing_slot {
            return self.inner.wait_as_follower(slot);
        }

        // Slow path: double-checked write lock to elect a single leader.
        let follower_slot: Arc<FlightSlot> = {
            let mut tbl = self.inner.table.write().unwrap_or_else(|e| e.into_inner());
            if let Some(&warm_until) = tbl.warm_until.get(&flight_key) {
                if now_ms < warm_until {
                    self.inner.warm_bypasses.fetch_add(1, Ordering::Relaxed);
                    return FlightRole::BypassWarm;
                }
            }
            if let Some(slot) = tbl.inflight.get(&flight_key) {
                Arc::clone(slot)
            } else {
                let slot = Arc::new(FlightSlot::new());
                tbl.inflight.insert(flight_key, Arc::clone(&slot));
                self.inner.leaders.fetch_add(1, Ordering::Relaxed);
                drop(tbl);
                return FlightRole::Leader(LeaderGuard {
                    inner: Arc::clone(&self.inner),
                    flight_key,
                    slot,
                    released: false,
                });
            }
        };

        self.inner.wait_as_follower(follower_slot)
    }

    /// Evicts a warm entry so the next request re-runs leader election.
    pub fn invalidate(&self, flight_key: u64) {
        let mut tbl = self.inner.table.write().unwrap_or_else(|e| e.into_inner());
        tbl.warm_until.remove(&flight_key);
    }

    /// Returns a snapshot of current coalescer telemetry counters.
    pub fn stats(&self) -> CoalescerStats {
        CoalescerStats {
            leaders: self.inner.leaders.load(Ordering::Relaxed),
            followers_coalesced: self.inner.followers_coalesced.load(Ordering::Relaxed),
            followers_warmed: self.inner.followers_warmed.load(Ordering::Relaxed),
            followers_timed_out: self.inner.followers_timed_out.load(Ordering::Relaxed),
            warm_bypasses: self.inner.warm_bypasses.load(Ordering::Relaxed),
            total_wait_ms: self.inner.total_wait_ms.load(Ordering::Relaxed),
        }
    }

    /// Returns the number of currently in-flight leaders.
    pub fn inflight_count(&self) -> usize {
        let tbl = self.inner.table.read().unwrap_or_else(|e| e.into_inner());
        tbl.inflight.len()
    }

    /// Returns the number of warm entries currently tracked.
    pub fn warm_count(&self) -> usize {
        let tbl = self.inner.table.read().unwrap_or_else(|e| e.into_inner());
        tbl.warm_until.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn test_32_thread_thundering_herd_coalesces_to_single_leader() {
        let coalescer = SingleFlightCoalescer::new(5_000, 60_000);
        let key = SingleFlightCoalescer::compute_flight_key(42, 0x1234, 0xdead_beef);
        let n_threads = 32usize;
        let barrier = Arc::new(Barrier::new(n_threads));
        let leader_count = Arc::new(AtomicUsize::new(0));
        let warmed_followers = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let c = coalescer.clone();
            let b = Arc::clone(&barrier);
            let lc = Arc::clone(&leader_count);
            let wf = Arc::clone(&warmed_followers);
            handles.push(thread::spawn(move || {
                b.wait();
                match c.acquire(key, 1_000) {
                    FlightRole::Leader(guard) => {
                        lc.fetch_add(1, Ordering::SeqCst);
                        // Wait deterministically until all 31 followers are parked
                        // (with a generous safety cap) to simulate leader TTFT.
                        let wait_start = Instant::now();
                        while c.stats().followers_coalesced < (n_threads as u64 - 1)
                            && wait_start.elapsed() < Duration::from_secs(3)
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                        thread::sleep(Duration::from_millis(15));
                        guard.mark_warm_and_release(1_050);
                    }
                    FlightRole::FollowerWaited {
                        waited_ms: _,
                        leader_warmed,
                    } => {
                        assert!(leader_warmed, "follower must observe leader_warmed = true");
                        wf.fetch_add(1, Ordering::SeqCst);
                    }
                    FlightRole::BypassWarm => {
                        panic!("cold burst should not observe BypassWarm before leader releases");
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        assert_eq!(leader_count.load(Ordering::SeqCst), 1);
        assert_eq!(warmed_followers.load(Ordering::SeqCst), 31);

        let st = coalescer.stats();
        assert_eq!(st.leaders, 1);
        assert_eq!(st.followers_coalesced, 31);
        assert_eq!(st.followers_warmed, 31);
        assert_eq!(st.followers_timed_out, 0);
        assert_eq!(st.warm_bypasses, 0);
        assert_eq!(coalescer.inflight_count(), 0);
        assert_eq!(coalescer.warm_count(), 1);
    }

    #[test]
    fn test_warm_bypass_and_ttl_expiration() {
        let coalescer = SingleFlightCoalescer::new(1_000, 5_000);
        let key = 0xabcd_ef01_u64;

        // First acquire elects leader.
        let guard = match coalescer.acquire(key, 10_000) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader, got {other:?}"),
        };
        guard.mark_warm_and_release(10_100); // warm until 15_100

        // Within TTL -> BypassWarm.
        for ts in [10_100, 12_000, 15_099] {
            match coalescer.acquire(key, ts) {
                FlightRole::BypassWarm => {}
                other => panic!("expected BypassWarm at {ts}, got {other:?}"),
            }
        }
        assert_eq!(coalescer.stats().warm_bypasses, 3);

        // At/past TTL -> expires and elects a new leader.
        let guard2 = match coalescer.acquire(key, 15_100) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader after TTL expiry, got {other:?}"),
        };
        guard2.mark_warm_and_release(15_150);
        assert_eq!(coalescer.stats().leaders, 2);

        // Explicit invalidation forces re-election even within TTL.
        coalescer.invalidate(key);
        match coalescer.acquire(key, 15_200) {
            FlightRole::Leader(g) => g.release_without_warming(),
            other => panic!("expected Leader after invalidate, got {other:?}"),
        }
    }

    #[test]
    fn test_leader_drop_fail_open() {
        let coalescer = SingleFlightCoalescer::new(5_000, 60_000);
        let key = 777;

        let guard = match coalescer.acquire(key, 100) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader, got {other:?}"),
        };

        let mut followers = Vec::new();
        for _ in 0..8 {
            let c = coalescer.clone();
            followers.push(thread::spawn(move || c.acquire(key, 105)));
        }

        // Wait until all 8 followers are coalesced on the in-flight leader.
        let t0 = Instant::now();
        while coalescer.stats().followers_coalesced < 8 && t0.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(1));
        }

        // Drop guard WITHOUT warming (simulates upstream 500 / disconnect).
        drop(guard);

        for h in followers {
            match h.join().expect("follower panicked") {
                FlightRole::FollowerWaited {
                    waited_ms: _,
                    leader_warmed,
                } => {
                    assert!(
                        !leader_warmed,
                        "dropped leader must signal leader_warmed = false"
                    );
                }
                other => panic!("expected FollowerWaited, got {other:?}"),
            }
        }

        let st = coalescer.stats();
        assert_eq!(st.leaders, 1);
        assert_eq!(st.followers_coalesced, 8);
        assert_eq!(st.followers_warmed, 0);
        assert_eq!(st.followers_timed_out, 0);
        assert_eq!(coalescer.inflight_count(), 0);
        assert_eq!(coalescer.warm_count(), 0);

        // Next caller must immediately become a new leader (not warm).
        match coalescer.acquire(key, 200) {
            FlightRole::Leader(g) => g.mark_warm_and_release(210),
            other => panic!("expected new Leader after failed leader drop, got {other:?}"),
        }
    }

    #[test]
    fn test_leader_panic_fail_open() {
        let coalescer = SingleFlightCoalescer::new(5_000, 60_000);
        let key = 999;

        let c_leader = coalescer.clone();
        let started = Arc::new(Barrier::new(2));
        let started_l = Arc::clone(&started);

        let leader_handle = thread::spawn(move || {
            let _guard = match c_leader.acquire(key, 500) {
                FlightRole::Leader(g) => g,
                other => panic!("expected Leader, got {other:?}"),
            };
            started_l.wait();
            let t0 = Instant::now();
            while c_leader.stats().followers_coalesced < 1
                && t0.elapsed() < Duration::from_secs(2)
            {
                thread::sleep(Duration::from_millis(1));
            }
            panic!("simulated leader panic before TTFT");
        });

        started.wait();
        let role = coalescer.acquire(key, 505);
        assert!(leader_handle.join().is_err());

        match role {
            FlightRole::FollowerWaited {
                leader_warmed: false,
                ..
            } => {}
            other => panic!("expected FollowerWaited(leader_warmed=false), got {other:?}"),
        }
        assert_eq!(coalescer.inflight_count(), 0);
    }

    #[test]
    fn test_follower_timeout_fail_open() {
        let coalescer = SingleFlightCoalescer::new(35, 60_000);
        let key = 404;

        let guard = match coalescer.acquire(key, 1_000) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader, got {other:?}"),
        };

        let c_follower = coalescer.clone();
        let t0 = Instant::now();
        let follower_handle = thread::spawn(move || c_follower.acquire(key, 1_005));

        match follower_handle.join().expect("follower panicked") {
            FlightRole::FollowerWaited {
                waited_ms,
                leader_warmed,
            } => {
                assert!(!leader_warmed);
                assert!(
                    waited_ms >= 25,
                    "expected waited_ms near 35ms, got {waited_ms}"
                );
            }
            other => panic!("expected timed-out FollowerWaited, got {other:?}"),
        }
        assert!(t0.elapsed() < Duration::from_secs(2));

        let st = coalescer.stats();
        assert_eq!(st.followers_coalesced, 1);
        assert_eq!(st.followers_timed_out, 1);
        assert_eq!(st.followers_warmed, 0);

        // Leader can still warm cleanly afterwards.
        guard.mark_warm_and_release(1_200);
        match coalescer.acquire(key, 1_250) {
            FlightRole::BypassWarm => {}
            other => panic!("expected BypassWarm after delayed leader finished, got {other:?}"),
        }
    }

    #[test]
    fn test_distinct_keys_independent() {
        let coalescer = SingleFlightCoalescer::new(1_000, 10_000);
        let g1 = match coalescer.acquire(1, 0) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader for key 1, got {other:?}"),
        };
        let g2 = match coalescer.acquire(2, 0) {
            FlightRole::Leader(g) => g,
            other => panic!("expected Leader for key 2, got {other:?}"),
        };
        assert_eq!(g1.flight_key(), 1);
        assert_eq!(g2.flight_key(), 2);
        assert_eq!(coalescer.inflight_count(), 2);
        g1.mark_warm_and_release(10);
        g2.mark_warm_and_release(10);
        assert_eq!(coalescer.inflight_count(), 0);
        assert_eq!(coalescer.warm_count(), 2);
    }
}
