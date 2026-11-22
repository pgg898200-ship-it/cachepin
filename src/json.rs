//! Zero-dependency RFC 8259 JSON parser, canonical serializer, and DOM helpers.
//!
//! Enforces a strict recursion depth limit (`MAX_DEPTH = 128`), validates all
//! UTF-8 and UTF-16 surrogate pairs, preserves exact numeric lexemes, and never
//! panics on malformed or adversarial inputs.

/// Maximum permitted container nesting depth for `parse_json`.
pub const MAX_DEPTH: usize = 128;

/// Lossless JSON value representation preserving object insertion order and
/// exact numeric lexemes.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    #[inline]
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries.as_slice()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_object_mut(&mut self) -> Option<&mut Vec<(String, JsonValue)>> {
        match self {
            JsonValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    #[inline]
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<JsonValue>> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(s) => s.parse::<f64>().ok().filter(|v| v.is_finite()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(s) => s.parse::<u64>().ok(),
            _ => None,
        }
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                entries.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            _ => None,
        }
    }

    /// Sets `key` to `val` if `self` is a `JsonValue::Object`.
    /// Updates the first existing entry if present (removing any stale duplicates),
    /// otherwise appends `(key.to_string(), val)`.
    pub fn set_field(&mut self, key: &str, val: JsonValue) {
        if let JsonValue::Object(entries) = self {
            let mut updated = false;
            let mut i = 0usize;
            let mut val_opt = Some(val);
            while i < entries.len() {
                if entries[i].0 == key {
                    if !updated {
                        if let Some(v) = val_opt.take() {
                            entries[i].1 = v;
                        }
                        updated = true;
                        i += 1;
                    } else {
                        entries.remove(i);
                    }
                } else {
                    i += 1;
                }
            }
            if !updated {
                if let Some(v) = val_opt.take() {
                    entries.push((key.to_string(), v));
                }
            }
        }
    }

    /// Removes `key` from `self` if it is a `JsonValue::Object`, returning the
    /// first removed value (and stripping any duplicate entries with the same key).
    pub fn remove_field(&mut self, key: &str) -> Option<JsonValue> {
        if let JsonValue::Object(entries) = self {
            let mut removed = None;
            let mut i = 0usize;
            while i < entries.len() {
                if entries[i].0 == key {
                    let (_, v) = entries.remove(i);
                    if removed.is_none() {
                        removed = Some(v);
                    }
                } else {
                    i += 1;
                }
            }
            removed
        } else {
            None
        }
    }

    /// Recursively sorts all object entries stably by key in place without
    /// reordering arrays. Returns `true` iff any object had keys out of sorted order.
    pub fn canonicalize_in_place(&mut self) -> bool {
        match self {
            JsonValue::Object(entries) => {
                let mut reordered = entries.windows(2).any(|w| w[0].0 > w[1].0);
                if reordered {
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                }
                for (_, v) in entries.iter_mut() {
                    if v.canonicalize_in_place() {
                        reordered = true;
                    }
                }
                reordered
            }
            JsonValue::Array(items) => {
                let mut reordered = false;
                for item in items.iter_mut() {
                    if item.canonicalize_in_place() {
                        reordered = true;
                    }
                }
                reordered
            }
            _ => false,
        }
    }
}

struct Parser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    #[inline]
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn skip_ws(&mut self) {
        let bytes = self.bytes;
        let len = bytes.len();
        let mut pos = self.pos;
        while pos < len {
            let b = bytes[pos];
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                pos += 1;
            } else {
                break;
            }
        }
        self.pos = pos;
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.skip_ws();
        let b = self
            .peek()
            .ok_or_else(|| format!("unexpected EOF at byte {}", self.pos))?;
        match b {
            b'"' => self.parse_string().map(JsonValue::String),
            b'{' => {
                if depth >= MAX_DEPTH {
                    return Err(format!(
                        "JSON nesting depth exceeds maximum of {MAX_DEPTH} at byte {}",
                        self.pos
                    ));
                }
                self.parse_object(depth + 1)
            }
            b'[' => {
                if depth >= MAX_DEPTH {
                    return Err(format!(
                        "JSON nesting depth exceeds maximum of {MAX_DEPTH} at byte {}",
                        self.pos
                    ));
                }
                self.parse_array(depth + 1)
            }
            b'-' | b'0'..=b'9' => self.parse_number(),
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'n' => self.parse_null(),
            other => Err(format!(
                "unexpected byte 0x{other:02x} at byte {}",
                self.pos
            )),
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.bytes.get(self.pos..self.pos + 4) == Some(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_true(&mut self) -> Result<JsonValue, String> {
        if self.bytes.get(self.pos..self.pos + 4) == Some(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_false(&mut self) -> Result<JsonValue, String> {
        if self.bytes.get(self.pos..self.pos + 5) == Some(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;

        if self.peek() == Some(b'-') {
            self.pos += 1;
        }

        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!("leading zero in JSON number at byte {start}"));
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at byte {start}")),
        }

        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!(
                    "expected digit after decimal point at byte {}",
                    self.pos
                ));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("expected digit in exponent at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        Ok(JsonValue::Number(self.src[start..self.pos].to_string()))
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let bytes = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| format!("truncated \\uXXXX escape at byte {}", self.pos))?;
        let mut val: u16 = 0;
        for &b in bytes {
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => {
                    return Err(format!(
                        "invalid hex digit 0x{b:02x} in \\uXXXX escape at byte {}",
                        self.pos
                    ));
                }
            };
            val = (val << 4) | digit;
        }
        self.pos += 4;
        Ok(val)
    }

    #[inline]
    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;

        let bytes = self.bytes;
        let len = bytes.len();
        let start = self.pos;

        // AVX2-accelerated fast path: scan unescaped string 32 bytes/iter with zero intermediate allocations.
        let pos = start + crate::simd::find_json_special_byte(&bytes[start..]);
        if pos >= len {
            return Err(format!("unterminated JSON string starting near byte {start}"));
        }
        let first_special = bytes[pos];
        if first_special == b'"' {
            self.pos = pos + 1;
            return Ok(self.src[start..pos].to_string());
        }
        if first_special < 0x20 {
            return Err(format!(
                "unescaped control character 0x{first_special:02x} in string at byte {pos}"
            ));
        }

        let mut out = String::with_capacity((pos - start) + 32);
        out.push_str(&self.src[start..pos]);
        self.pos = pos;
        let mut chunk_start = pos;

        while self.pos < len {
            let rel = crate::simd::find_json_special_byte(&bytes[self.pos..]);
            self.pos += rel;
            if self.pos >= len {
                break;
            }
            let b = bytes[self.pos];
            match b {
                b'"' => {
                    if chunk_start < self.pos {
                        out.push_str(&self.src[chunk_start..self.pos]);
                    }
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    if chunk_start < self.pos {
                        out.push_str(&self.src[chunk_start..self.pos]);
                    }
                    self.pos += 1;

                    let esc = self
                        .peek()
                        .ok_or_else(|| format!("unterminated escape at byte {}", self.pos))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(format!(
                                        "unpaired high surrogate \\u{code:04X} at byte {}",
                                        self.pos
                                    ));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(format!(
                                        "invalid low surrogate \\u{low:04X} after \\u{code:04X} at byte {}",
                                        self.pos
                                    ));
                                }
                                let scalar = 0x10000
                                    + ((((code as u32) - 0xD800) << 10) | ((low as u32) - 0xDC00));
                                let ch = char::from_u32(scalar).ok_or_else(|| {
                                    format!("invalid unicode scalar U+{scalar:04X}")
                                })?;
                                out.push(ch);
                            } else if (0xDC00..=0xDFFF).contains(&code) {
                                return Err(format!(
                                    "unexpected lone low surrogate \\u{code:04X} at byte {}",
                                    self.pos
                                ));
                            } else {
                                let ch = char::from_u32(code as u32).ok_or_else(|| {
                                    format!("invalid unicode scalar U+{code:04X}")
                                })?;
                                out.push(ch);
                            }
                        }
                        other => {
                            return Err(format!(
                                "invalid escape sequence '\\x{other:02x}' at byte {}",
                                self.pos
                            ));
                        }
                    }
                    chunk_start = self.pos;
                }
                _ => {
                    return Err(format!(
                        "unescaped control character 0x{b:02x} in string at byte {}",
                        self.pos
                    ));
                }
            }
        }

        Err(format!("unterminated JSON string starting near byte {start}"))
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // consume '['
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(Vec::new()));
        }
        let mut items = Vec::with_capacity(4);

        loop {
            let val = self.parse_value(depth)?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                Some(other) => {
                    return Err(format!(
                        "expected ',' or ']' in array, found 0x{other:02x} at byte {}",
                        self.pos
                    ));
                }
                None => return Err(format!("unterminated array at byte {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // consume '{'
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(Vec::new()));
        }
        let mut entries = Vec::with_capacity(4);

        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!(
                    "expected string key in object at byte {}",
                    self.pos
                ));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!(
                    "expected ':' after object key at byte {}",
                    self.pos
                ));
            }
            self.pos += 1;
            let val = self.parse_value(depth)?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                Some(other) => {
                    return Err(format!(
                        "expected ',' or '}}' in object, found 0x{other:02x} at byte {}",
                        self.pos
                    ));
                }
                None => return Err(format!("unterminated object at byte {}", self.pos)),
            }
        }
    }
}

/// Parses an RFC 8259 JSON document from UTF-8 bytes.
pub fn parse_json(input: &[u8]) -> Result<JsonValue, String> {
    let src = std::str::from_utf8(input)
        .map_err(|e| format!("invalid UTF-8 in JSON input at byte {}", e.valid_up_to()))?;
    let mut parser = Parser::new(src);
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != input.len() {
        return Err(format!(
            "trailing non-whitespace bytes at byte {}",
            parser.pos
        ));
    }
    Ok(val)
}

#[inline]
fn write_escaped_string(s: &str, out: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = s.as_bytes();
    out.reserve(bytes.len() + 2);
    out.push(b'"');
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let rel = crate::simd::find_json_special_byte(&bytes[i..]);
        let hit = i + rel;
        if hit == len {
            out.extend_from_slice(&bytes[i..]);
            break;
        }
        if hit > i {
            out.extend_from_slice(&bytes[i..hit]);
        }
        let b = bytes[hit];
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0C => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            _ => out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[(b >> 4) as usize],
                HEX[(b & 0x0F) as usize],
            ]),
        }
        i = hit + 1;
    }

    out.push(b'"');
}

/// Serializes `val` as compact RFC 8259 JSON preserving existing object key order.
pub fn serialize_json(val: &JsonValue, out: &mut Vec<u8>) {
    match val {
        JsonValue::Null => out.extend_from_slice(b"null"),
        JsonValue::Bool(true) => out.extend_from_slice(b"true"),
        JsonValue::Bool(false) => out.extend_from_slice(b"false"),
        JsonValue::Number(n) => out.extend_from_slice(n.as_bytes()),
        JsonValue::String(s) => write_escaped_string(s, out),
        JsonValue::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serialize_json(item, out);
            }
            out.push(b']');
        }
        JsonValue::Object(entries) => {
            out.push(b'{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_escaped_string(k, out);
                out.push(b':');
                serialize_json(v, out);
            }
            out.push(b'}');
        }
    }
}

/// Serializes `val` as canonical RFC 8259 JSON by stably sorting every object's
/// entries lexicographically by key without reordering arrays.
///
/// Returns `true` iff any object (root or nested) had keys out of sorted order.
pub fn serialize_canonical(val: &JsonValue, out: &mut Vec<u8>) -> bool {
    match val {
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
            serialize_json(val, out);
            false
        }
        JsonValue::Array(items) => {
            let mut reordered = false;
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                if serialize_canonical(item, out) {
                    reordered = true;
                }
            }
            out.push(b']');
            reordered
        }
        JsonValue::Object(entries) => {
            let unsorted = entries.windows(2).any(|w| w[0].0 > w[1].0);
            let mut reordered = unsorted;
            out.push(b'{');
            if !unsorted {
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write_escaped_string(k, out);
                    out.push(b':');
                    if serialize_canonical(v, out) {
                        reordered = true;
                    }
                }
            } else {
                let mut order: Vec<usize> = (0..entries.len()).collect();
                order.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
                for (i, idx) in order.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    let (ref k, ref v) = entries[idx];
                    write_escaped_string(k, out);
                    out.push(b':');
                    if serialize_canonical(v, out) {
                        reordered = true;
                    }
                }
            }
            out.push(b'}');
            reordered
        }
    }
}

/// Zero-clone canonical serialization of `val` omitting any top-level key equal to `skip_top_key`.
pub fn serialize_canonical_skip_top_key(val: &JsonValue, skip_top_key: &str, out: &mut Vec<u8>) {
    match val {
        JsonValue::Object(entries) => {
            let unsorted = entries.windows(2).any(|w| w[0].0 > w[1].0);
            out.push(b'{');
            let mut emitted = false;
            if !unsorted {
                for (k, v) in entries {
                    if k == skip_top_key {
                        continue;
                    }
                    if emitted {
                        out.push(b',');
                    }
                    write_escaped_string(k, out);
                    out.push(b':');
                    let _ = serialize_canonical(v, out);
                    emitted = true;
                }
            } else {
                let mut order: Vec<usize> = (0..entries.len())
                    .filter(|&i| entries[i].0 != skip_top_key)
                    .collect();
                order.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
                for idx in order {
                    if emitted {
                        out.push(b',');
                    }
                    let (ref k, ref v) = entries[idx];
                    write_escaped_string(k, out);
                    out.push(b':');
                    let _ = serialize_canonical(v, out);
                    emitted = true;
                }
            }
            out.push(b'}');
        }
        _ => {
            let _ = serialize_canonical(val, out);
        }
    }
}

/// Zero-clone compact serialization of an already-canonicalized `val` omitting any top-level
/// key equal to `skip_top_key`.
#[inline]
pub fn serialize_json_skip_top_key(val: &JsonValue, skip_top_key: &str, out: &mut Vec<u8>) {
    match val {
        JsonValue::Object(entries) => {
            out.push(b'{');
            let mut emitted = false;
            for (k, v) in entries {
                if k == skip_top_key {
                    continue;
                }
                if emitted {
                    out.push(b',');
                }
                write_escaped_string(k, out);
                out.push(b':');
                serialize_json(v, out);
                emitted = true;
            }
            out.push(b'}');
        }
        _ => serialize_json(val, out),
    }
}

/// Serializes `val` into an owned compact JSON `String`.
pub fn json_to_string(val: &JsonValue) -> String {
    let mut out = Vec::new();
    serialize_json(val, &mut out);
    // All bytes emitted by `serialize_json` are guaranteed valid UTF-8.
    String::from_utf8(out).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primitives_and_roundtrip() {
        assert_eq!(parse_json(b"null").unwrap(), JsonValue::Null);
        assert_eq!(parse_json(b"  true \n\t").unwrap(), JsonValue::Bool(true));
        assert_eq!(parse_json(b"false").unwrap(), JsonValue::Bool(false));
        assert_eq!(
            parse_json(b"-0.125e+10").unwrap(),
            JsonValue::Number("-0.125e+10".to_string())
        );
        assert_eq!(
            parse_json(b"0").unwrap(),
            JsonValue::Number("0".to_string())
        );
        assert_eq!(
            parse_json(b"\"hello\\n\\t\\\"\\\\\\/\\b\\f\\r\"").unwrap(),
            JsonValue::String("hello\n\t\"\\/\u{0008}\u{000C}\r".to_string())
        );

        let complex = br#"{"z":[1,true,null,"ok",{"b":"x","a":"\u0000\u001f"}],"a":-42.5}"#;
        let parsed = parse_json(complex).unwrap();
        let serialized = json_to_string(&parsed);
        let reparsed = parse_json(serialized.as_bytes()).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn test_utf16_escapes_and_surrogate_pairs() {
        // Basic BMP escape + CJK + surrogate pair for U+1F600 (😀) and U+1D11E (𝄞).
        let raw = br#""\u0041\u4e2d\uD83D\uDE00\uD834\uDD1E""#;
        let val = parse_json(raw).unwrap();
        assert_eq!(val.as_str(), Some("A中😀𝄞"));

        // Re-serialize and parse back cleanly.
        let s = json_to_string(&val);
        assert_eq!(parse_json(s.as_bytes()).unwrap(), val);

        // Adversarial malformed surrogates must return Err, never panic.
        assert!(parse_json(br#""\uD800""#).is_err());
        assert!(parse_json(br#""\uD800abc""#).is_err());
        assert!(parse_json(br#""\uD800\u0041""#).is_err());
        assert!(parse_json(br#""\uD800\uD800""#).is_err());
        assert!(parse_json(br#""\uDC00""#).is_err());
        assert!(parse_json(br#""\uDFFF""#).is_err());
    }

    #[test]
    fn test_rfc8259_rejects_malformed_inputs() {
        let bad_inputs: &[&[u8]] = &[
            b"",
            b"   \t\r\n",
            b"true false",
            b"01",
            b"-01",
            b"+1",
            b".5",
            b"1.",
            b"1e",
            b"1e+",
            b"-",
            b"[1,]",
            b"{\"a\":1,}",
            b"{\"a\" 1}",
            b"{1:2}",
            b"\"unterminated",
            b"\"\n\"",
            b"\"\x00\"",
            b"\"\\x41\"",
            b"\"\\u123\"",
            b"\"\\u123g\"",
            b"[1, 2",
            b"{\"a\": 1",
            &[b'"', 0xFF, 0xFE, b'"'],
        ];
        for bad in bad_inputs {
            assert!(
                parse_json(bad).is_err(),
                "expected Err for malformed JSON {:?}",
                bad
            );
        }
    }

    #[test]
    fn test_depth_cap_prevents_stack_overflow() {
        // Exactly MAX_DEPTH nested arrays should parse cleanly.
        let mut ok_arr = Vec::new();
        ok_arr.extend(std::iter::repeat(b'[').take(MAX_DEPTH));
        ok_arr.push(b'1');
        ok_arr.extend(std::iter::repeat(b']').take(MAX_DEPTH));
        assert!(parse_json(&ok_arr).is_ok());

        // MAX_DEPTH + 1 nested arrays must fail cleanly.
        let mut too_deep_arr = Vec::new();
        too_deep_arr.extend(std::iter::repeat(b'[').take(MAX_DEPTH + 1));
        too_deep_arr.push(b'1');
        too_deep_arr.extend(std::iter::repeat(b']').take(MAX_DEPTH + 1));
        assert!(parse_json(&too_deep_arr).is_err());

        // Deeply nested objects (e.g. 256 levels) must fail cleanly.
        let mut deep_obj = String::new();
        for _ in 0..256 {
            deep_obj.push_str("{\"a\":");
        }
        deep_obj.push_str("null");
        for _ in 0..256 {
            deep_obj.push('}');
        }
        assert!(parse_json(deep_obj.as_bytes()).is_err());
    }

    #[test]
    fn test_serialize_canonical_sorting_and_array_preservation() {
        // Already sorted -> returns false.
        let sorted = parse_json(br#"{"a":1,"b":[3,1,2],"c":{"x":1,"y":2}}"#).unwrap();
        let mut out = Vec::new();
        assert!(!serialize_canonical(&sorted, &mut out));
        assert_eq!(out, br#"{"a":1,"b":[3,1,2],"c":{"x":1,"y":2}}"#);

        // Unsorted nested inside an array -> returns true, preserves array order.
        let nested_unsorted =
            parse_json(br#"{"a":1,"b":[3,{"z":1,"a":2},1]}"#).unwrap();
        let mut out2 = Vec::new();
        assert!(serialize_canonical(&nested_unsorted, &mut out2));
        assert_eq!(out2, br#"{"a":1,"b":[3,{"a":2,"z":1},1]}"#);

        // Stable sort on duplicate keys preserves their relative order.
        let dup = JsonValue::Object(vec![
            ("b".to_string(), JsonValue::Number("1".to_string())),
            ("a".to_string(), JsonValue::Number("2".to_string())),
            ("b".to_string(), JsonValue::Number("3".to_string())),
        ]);
        let mut out3 = Vec::new();
        assert!(serialize_canonical(&dup, &mut out3));
        assert_eq!(out3, br#"{"a":2,"b":1,"b":3}"#);
    }

    #[test]
    fn test_json_value_convenience_methods() {
        let mut obj = parse_json(
            br#"{"s":"hi","n":42,"f":3.5,"b":true,"arr":[1,2],"inner":{"k":"v"}}"#,
        )
        .unwrap();

        assert_eq!(obj.get("s").and_then(JsonValue::as_str), Some("hi"));
        assert_eq!(obj.get("n").and_then(JsonValue::as_u64), Some(42));
        assert_eq!(obj.get("f").and_then(JsonValue::as_f64), Some(3.5));
        assert_eq!(obj.get("b").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(obj.get("arr").and_then(JsonValue::as_array).map(|a| a.len()), Some(2));
        assert!(obj.as_object().is_some());

        obj.get_mut("arr")
            .and_then(JsonValue::as_array_mut)
            .unwrap()
            .push(JsonValue::Number("3".to_string()));
        assert_eq!(obj.get("arr").and_then(JsonValue::as_array).map(|a| a.len()), Some(3));

        obj.set_field("n", JsonValue::Number("99".to_string()));
        assert_eq!(obj.get("n").and_then(JsonValue::as_u64), Some(99));

        obj.set_field("added", JsonValue::Bool(false));
        assert_eq!(obj.get("added").and_then(JsonValue::as_bool), Some(false));

        let removed = obj.remove_field("added");
        assert_eq!(removed, Some(JsonValue::Bool(false)));
        assert!(obj.get("added").is_none());
        assert!(obj.as_object_mut().is_some());
    }
}
