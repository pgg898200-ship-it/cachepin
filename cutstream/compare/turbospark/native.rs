mod stop_matcher;
use std::{env, fs, time::Instant};
use stop_matcher::StreamingStopMatcher;

fn absorb(hash: &mut u64, bytes: &mut usize, output: String, stopped: bool) {
    for &c in output.as_bytes() {
        *hash = (*hash ^ u64::from(c)).wrapping_mul(1099511628211);
    }
    *hash = (*hash ^ (256 + stopped as u64)).wrapping_mul(1099511628211);
    *bytes += output.len();
}
fn main() {
    let args: Vec<_> = env::args().collect();
    let budget_ns: u128 = args[2].parse().unwrap();
    let stops: Vec<String> = fs::read_to_string(format!("{}/stops.txt", args[1]))
        .unwrap().lines().map(str::to_owned).collect();
    let raw = fs::read_to_string(format!("{}/events.tsv", args[1])).unwrap();
    let events: Vec<(usize, &str)> = raw.lines().map(|line| {
        let (id, text) = line.split_once('\t').unwrap();
        (id.parse().unwrap(), text)
    }).collect();
    let streams = events.iter().map(|e| e.0 + 1).max().unwrap();
    let mut elapsed = 0;
    let mut runs = 0;
    let mut expected = None;
    loop {
        let mut matchers: Vec<_> = (0..streams)
            .map(|_| StreamingStopMatcher::new(stops.clone())).collect();
        let mut hash = 14695981039346656037;
        let mut bytes = 0;
        let start = Instant::now();
        for &(id, text) in &events {
            let matcher = &mut matchers[id];
            let output = matcher.push(text);
            absorb(&mut hash, &mut bytes, output, matcher.is_stopped());
        }
        for matcher in &mut matchers {
            let stopped_on_marker = matcher.is_stopped();
            absorb(&mut hash, &mut bytes, matcher.finish(), stopped_on_marker);
        }
        elapsed += start.elapsed().as_nanos();
        runs += 1;
        if let Some(prior) = expected { assert_eq!(prior, (hash, bytes)); }
        expected = Some((hash, bytes));
        if elapsed >= budget_ns { break; }
    }
    let (hash, bytes) = expected.unwrap();
    println!("{hash} {bytes} {elapsed} {runs}");
}
