//! The built-in node library.
//!
//! Each [`Spec`] names its inputs and outputs and evaluates one call. Inputs
//! not marked `list` are scalar: a list arriving there makes the node run once
//! per element (see [`crate::value::lace`]), so `Add` over two lists adds them
//! pairwise without any list-aware code in `Add` itself. Angles are degrees.
//!
//! An application adds its own nodes as further [`Spec`]s whose
//! [`Eval::Host`] hands the call to [`Host::call`].

use std::cmp::Ordering;

use serde_json::Value;

use crate::value::{boolean, list, num, number, point, point_value, text};
use crate::Host;
use Eval::Pure;

/// An input's value while nothing is linked to it.
#[derive(Clone, Copy, Debug)]
pub enum Initial {
    None,
    Num(f64),
    Text(&'static str),
    Bool(bool),
}

impl Initial {
    pub fn value(self) -> Value {
        match self {
            Initial::None => Value::Null,
            Initial::Num(x) => number(x),
            Initial::Text(s) => Value::String(s.to_owned()),
            Initial::Bool(b) => Value::Bool(b),
        }
    }
}

#[derive(Debug)]
pub struct Input {
    pub name: &'static str,
    pub default: Initial,
    /// Takes a whole list instead of running once per element.
    pub list: bool,
}

/// How a node computes its outputs from one set of inputs.
#[derive(Clone, Copy)]
pub enum Eval {
    /// A pure function of the inputs.
    Pure(fn(&[Value], &dyn Host) -> Vec<Value>),
    /// The host computes it, see [`Host::call`].
    Host,
}

pub struct Spec {
    pub name: &'static str,
    pub category: &'static str,
    pub inputs: &'static [Input],
    pub outputs: &'static [&'static str],
    pub eval: Eval,
}

impl std::fmt::Debug for Spec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

pub const CATEGORIES: &[&str] = &["Input", "Math", "Logic", "List", "Text", "Geometry"];

/// The library spec called `name`.
pub fn spec(name: &str) -> Option<&'static Spec> {
    LIBRARY.iter().find(|spec| spec.name == name)
}

pub const fn n(name: &'static str, x: f64) -> Input {
    Input { name, default: Initial::Num(x), list: false }
}
pub const fn s(name: &'static str, value: &'static str) -> Input {
    Input { name, default: Initial::Text(value), list: false }
}
pub const fn b(name: &'static str, value: bool) -> Input {
    Input { name, default: Initial::Bool(value), list: false }
}
/// A scalar input with no default (points, curves, arbitrary items).
pub const fn v(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: false }
}
/// A whole-list input.
pub const fn l(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: true }
}

const X: &[Input] = &[n("x", 0.0)];
const AB: &[Input] = &[n("a", 0.0), n("b", 0.0)];
const PQ: &[Input] = &[v("a"), v("b")];
const LIST: &[Input] = &[l("list")];
const TEXT: &[Input] = &[s("text", "")];
const RESULT: &[&str] = &["result"];
const VALUE: &[&str] = &["value"];
const POINT: &[&str] = &["point"];

fn out(value: Value) -> Vec<Value> {
    vec![value]
}

fn f1(a: &[Value], f: fn(f64) -> f64) -> Vec<Value> {
    out(num(&a[0]).map_or(Value::Null, |x| number(f(x))))
}

fn f2(a: &[Value], f: fn(f64, f64) -> f64) -> Vec<Value> {
    out(match (num(&a[0]), num(&a[1])) {
        (Some(x), Some(y)) => number(f(x, y)),
        _ => Value::Null,
    })
}

fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => x.partial_cmp(&y),
        _ => Some(text(a).cmp(&text(b))),
    }
}

fn equal(a: &Value, b: &Value) -> bool {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => a == b,
    }
}

fn test(a: &[Value], f: fn(Ordering) -> bool) -> Vec<Value> {
    out(compare(&a[0], &a[1]).map_or(Value::Null, |o| Value::Bool(f(o))))
}

fn logic(a: &[Value], f: fn(bool, bool) -> bool) -> Vec<Value> {
    out(match (boolean(&a[0]), boolean(&a[1])) {
        (Some(x), Some(y)) => Value::Bool(f(x, y)),
        _ => Value::Null,
    })
}

fn numbers(value: &Value) -> Vec<f64> {
    list(value).iter().filter_map(num).collect()
}

fn index(len: usize, i: f64) -> Option<usize> {
    let i = i as i64;
    let i = if i < 0 { len as i64 + i } else { i };
    (0..len as i64).contains(&i).then_some(i as usize)
}

/// Upper bound on generated list lengths, so a mistyped step cannot hang the
/// evaluation.
const MAX_ITEMS: usize = 100_000;

fn range(start: f64, count: usize, step: f64) -> Value {
    Value::Array((0..count.min(MAX_ITEMS)).map(|k| number(start + step * k as f64)).collect())
}

fn flatten(value: &Value, into: &mut Vec<Value>) {
    match value {
        Value::Array(items) => items.iter().for_each(|item| flatten(item, into)),
        other => into.push(other.clone()),
    }
}

fn map_point(a: &Value, f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<Value> {
    out(point(a).map_or(Value::Null, |p| point_value(f(p))))
}

fn rotate_xy(p: [f64; 3], c: [f64; 3], degrees: f64) -> [f64; 3] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    [c[0] + dx * cos - dy * sin, c[1] + dx * sin + dy * cos, p[2]]
}

/// Replaces the variables `a`..`d` in a formula with their values.
fn substitute(expression: &str, values: &[Value]) -> String {
    let mut result = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, result: &mut String| {
        match ["a", "b", "c", "d"].iter().position(|name| *name == word.as_str()) {
            Some(i) => result.push_str(&format!("({})", num(&values[i]).unwrap_or(0.0))),
            None => result.push_str(word),
        }
        word.clear();
    };
    for ch in expression.chars() {
        if ch.is_alphanumeric() || ch == '_' || (ch == '.' && !word.is_empty()) {
            word.push(ch);
        } else {
            flush(&mut word, &mut result);
            result.push(ch);
        }
    }
    flush(&mut word, &mut result);
    result
}

/// A seeded xorshift stream in `[0, 1)`.
fn random(seed: f64) -> impl FnMut() -> f64 {
    let mut state = (seed as u64) ^ 0x9E37_79B9_7F4A_7C15;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A count input: a non-negative whole number, capped.
fn amount(value: &Value) -> usize {
    (num(value).unwrap_or(0.0).max(0.0) as usize).min(MAX_ITEMS)
}

/// The distinct items of `a` that `keep` accepts by whether `b` has them,
/// then, for a union, the rest of `b`.
fn set(a: &Value, b: &Value, union: bool, keep: fn(bool) -> bool) -> Vec<Value> {
    let other = list(b);
    let mut items: Vec<Value> = Vec::new();
    let mut push = |item: Value| {
        if !items.iter().any(|seen| equal(seen, &item)) {
            items.push(item);
        }
    };
    for item in list(a) {
        if keep(other.iter().any(|x| equal(x, &item))) {
            push(item);
        }
    }
    if union {
        other.into_iter().for_each(push);
    }
    items
}

/// The items of `a` ordered by the keys in `b`, with the keys in that order.
fn sort_by_keys(a: &Value, b: &Value) -> (Vec<Value>, Vec<Value>) {
    let mut pairs: Vec<(Value, Value)> = list(a).into_iter().zip(list(b)).collect();
    pairs.sort_by(|x, y| compare(&x.1, &y.1).unwrap_or(Ordering::Equal));
    pairs.into_iter().unzip()
}

/// The `k`-item combinations of `items`, in order.
fn combinations(items: &[Value], k: usize, start: usize, current: &mut Vec<Value>, into: &mut Vec<Value>) {
    if current.len() == k {
        into.push(Value::Array(current.clone()));
        return;
    }
    for i in start..items.len() {
        if into.len() >= MAX_ITEMS {
            return;
        }
        current.push(items[i].clone());
        combinations(items, k, i + 1, current, into);
        current.pop();
    }
}

fn vectors(a: &[Value], f: fn([f64; 3], [f64; 3]) -> Value) -> Vec<Value> {
    out(match (point(&a[0]), point(&a[1])) {
        (Some(p), Some(q)) => f(p, q),
        _ => Value::Null,
    })
}

fn dot(p: [f64; 3], q: [f64; 3]) -> f64 {
    p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
}

fn cross(p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
    [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]
}

fn length(p: [f64; 3]) -> f64 {
    dot(p, p).sqrt()
}

pub const LIBRARY: &[Spec] = &[
    // ── Input ────────────────────────────────────────────────────────────
    Spec { name: "Number", category: "Input", inputs: &[n("value", 0.0)], outputs: VALUE,
        eval: Pure(|a, _| out(num(&a[0]).map_or(Value::Null, number))) },
    Spec { name: "Number Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 0.1)], outputs: VALUE,
        eval: Pure(|a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        }) },
    Spec { name: "Integer Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 1.0)], outputs: VALUE,
        eval: Pure(|a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0).round());
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        }) },
    Spec { name: "Text", category: "Input", inputs: &[s("value", "")], outputs: VALUE,
        eval: Pure(|a, _| out(Value::String(text(&a[0])))) },
    Spec { name: "Boolean", category: "Input", inputs: &[b("value", false)], outputs: VALUE,
        eval: Pure(|a, _| out(boolean(&a[0]).map_or(Value::Null, Value::Bool))) },
    Spec { name: "Formula", category: "Input",
        inputs: &[s("expression", "a+b"), n("a", 0.0), n("b", 0.0), n("c", 0.0), n("d", 0.0)],
        outputs: RESULT,
        eval: Pure(|a, host| out(host.expression(&substitute(&text(&a[0]), &a[1..])).map_or(Value::Null, number))) },
    Spec { name: "Watch", category: "Input", inputs: &[l("in")], outputs: &["out"],
        eval: Pure(|a, _| out(a[0].clone())) },
    // ── Math ─────────────────────────────────────────────────────────────
    Spec { name: "Add", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, |x, y| x + y)) },
    Spec { name: "Subtract", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, |x, y| x - y)) },
    Spec { name: "Multiply", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, |x, y| x * y)) },
    Spec { name: "Divide", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, |x, y| x / y)) },
    Spec { name: "Modulo", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, |x, y| x.rem_euclid(y))) },
    Spec { name: "Power", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, f64::powf)) },
    Spec { name: "Minimum", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, f64::min)) },
    Spec { name: "Maximum", category: "Math", inputs: AB, outputs: RESULT, eval: Pure(|a, _| f2(a, f64::max)) },
    Spec { name: "Negate", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| -x)) },
    Spec { name: "Absolute", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::abs)) },
    Spec { name: "Square Root", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::sqrt)) },
    Spec { name: "Exponential", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::exp)) },
    Spec { name: "Natural Log", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::ln)) },
    Spec { name: "Log10", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::log10)) },
    Spec { name: "Sine", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.to_radians().sin())) },
    Spec { name: "Cosine", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.to_radians().cos())) },
    Spec { name: "Tangent", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.to_radians().tan())) },
    Spec { name: "Arcsine", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.asin().to_degrees())) },
    Spec { name: "Arccosine", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.acos().to_degrees())) },
    Spec { name: "Arctangent", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, |x| x.atan().to_degrees())) },
    Spec { name: "Arctangent2", category: "Math", inputs: &[n("y", 0.0), n("x", 1.0)], outputs: RESULT,
        eval: Pure(|a, _| f2(a, |y, x| y.atan2(x).to_degrees())) },
    Spec { name: "Round", category: "Math", inputs: &[n("x", 0.0), n("digits", 0.0)], outputs: RESULT,
        eval: Pure(|a, _| f2(a, |x, d| { let m = 10f64.powi(d as i32); (x * m).round() / m })) },
    Spec { name: "Floor", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::floor)) },
    Spec { name: "Ceiling", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::ceil)) },
    Spec { name: "Clamp", category: "Math", inputs: &[n("x", 0.0), n("min", 0.0), n("max", 1.0)], outputs: RESULT,
        eval: Pure(|a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(lo), Some(hi)) => number(x.clamp(lo.min(hi), hi.max(lo))),
            _ => Value::Null,
        })) },
    Spec { name: "Remap", category: "Math",
        inputs: &[n("x", 0.0), n("from min", 0.0), n("from max", 1.0), n("to min", 0.0), n("to max", 1.0)],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let values: Option<Vec<f64>> = a.iter().map(num).collect();
            out(values.map_or(Value::Null, |v| number(v[3] + (v[0] - v[1]) / (v[2] - v[1]) * (v[4] - v[3]))))
        }) },
    Spec { name: "To Radians", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::to_radians)) },
    Spec { name: "To Degrees", category: "Math", inputs: X, outputs: RESULT, eval: Pure(|a, _| f1(a, f64::to_degrees)) },
    Spec { name: "Pi", category: "Math", inputs: &[], outputs: RESULT, eval: Pure(|_, _| out(number(std::f64::consts::PI))) },
    Spec { name: "E", category: "Math", inputs: &[], outputs: RESULT, eval: Pure(|_, _| out(number(std::f64::consts::E))) },
    Spec { name: "Random", category: "Math", inputs: &[n("seed", 1.0), n("count", 10.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let mut next = random(num(&a[0]).unwrap_or(1.0));
            out(Value::Array((0..amount(&a[1])).map(|_| number(next())).collect()))
        }) },
    Spec { name: "Sign", category: "Math", inputs: X, outputs: RESULT,
        eval: Pure(|a, _| f1(a, |x| if x == 0.0 { 0.0 } else { x.signum() })) },
    Spec { name: "Factorial", category: "Math", inputs: X, outputs: RESULT,
        eval: Pure(|a, _| f1(a, |x| if x < 0.0 || x.fract() != 0.0 {
            f64::NAN
        } else {
            (1..=x.min(171.0) as u64).map(|k| k as f64).product()
        })) },
    // ── Logic ────────────────────────────────────────────────────────────
    Spec { name: "Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(equal(&a[0], &a[1])))) },
    Spec { name: "Not Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(!equal(&a[0], &a[1])))) },
    Spec { name: "Less", category: "Logic", inputs: PQ, outputs: RESULT, eval: Pure(|a, _| test(a, Ordering::is_lt)) },
    Spec { name: "Less Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: Pure(|a, _| test(a, Ordering::is_le)) },
    Spec { name: "Greater", category: "Logic", inputs: PQ, outputs: RESULT, eval: Pure(|a, _| test(a, Ordering::is_gt)) },
    Spec { name: "Greater Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: Pure(|a, _| test(a, Ordering::is_ge)) },
    Spec { name: "And", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: Pure(|a, _| logic(a, |x, y| x && y)) },
    Spec { name: "Or", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: Pure(|a, _| logic(a, |x, y| x || y)) },
    Spec { name: "Xor", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: Pure(|a, _| logic(a, |x, y| x != y)) },
    Spec { name: "Not", category: "Logic", inputs: &[b("a", false)], outputs: RESULT,
        eval: Pure(|a, _| out(boolean(&a[0]).map_or(Value::Null, |x| Value::Bool(!x)))) },
    Spec { name: "If", category: "Logic", inputs: &[b("test", true), v("then"), v("else")], outputs: RESULT,
        eval: Pure(|a, _| out(match boolean(&a[0]) {
            Some(true) => a[1].clone(),
            Some(false) => a[2].clone(),
            None => Value::Null,
        })) },
    Spec { name: "Is Null", category: "Logic", inputs: &[l("value")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(a[0].is_null()))) },
    // ── List ─────────────────────────────────────────────────────────────
    Spec { name: "Range", category: "List", inputs: &[n("start", 0.0), n("end", 10.0), n("step", 1.0)],
        outputs: &["list"],
        eval: Pure(|a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(end), Some(step)) if step != 0.0 && (end - start) / step >= 0.0 => {
                range(start, ((end - start) / step + 1e-9).floor() as usize + 1, step)
            }
            _ => Value::Array(Vec::new()),
        })) },
    Spec { name: "Sequence", category: "List", inputs: &[n("start", 0.0), n("count", 10.0), n("step", 1.0)],
        outputs: &["list"],
        eval: Pure(|a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(count), Some(step)) => range(start, count.max(0.0) as usize, step),
            _ => Value::Array(Vec::new()),
        })) },
    Spec { name: "Create List", category: "List",
        inputs: &[l("item 0"), l("item 1"), l("item 2"), l("item 3"), l("item 4")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(a.iter().filter(|item| !item.is_null()).cloned().collect()))) },
    Spec { name: "Count", category: "List", inputs: LIST, outputs: &["count"],
        eval: Pure(|a, _| out(number(list(&a[0]).len() as f64))) },
    Spec { name: "Get Item", category: "List", inputs: &[l("list"), n("index", 0.0)], outputs: &["item"],
        eval: Pure(|a, _| {
            let items = list(&a[0]);
            out(num(&a[1]).and_then(|i| index(items.len(), i)).map_or(Value::Null, |i| items[i].clone()))
        }) },
    Spec { name: "First Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: Pure(|a, _| out(list(&a[0]).first().cloned().unwrap_or(Value::Null))) },
    Spec { name: "Last Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: Pure(|a, _| out(list(&a[0]).last().cloned().unwrap_or(Value::Null))) },
    Spec { name: "Reverse", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| { let mut items = list(&a[0]); items.reverse(); out(Value::Array(items)) }) },
    Spec { name: "Sort", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items = list(&a[0]);
            items.sort_by(|x, y| compare(x, y).unwrap_or(Ordering::Equal));
            out(Value::Array(items))
        }) },
    Spec { name: "Sum", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(number(numbers(&a[0]).iter().sum()))) },
    Spec { name: "Average", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| { let x = numbers(&a[0]); out(number(x.iter().sum::<f64>() / x.len() as f64)) }) },
    Spec { name: "List Minimum", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(numbers(&a[0]).into_iter().reduce(f64::min).map_or(Value::Null, number))) },
    Spec { name: "List Maximum", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(numbers(&a[0]).into_iter().reduce(f64::max).map_or(Value::Null, number))) },
    Spec { name: "Flatten", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| { let mut items = Vec::new(); flatten(&a[0], &mut items); out(Value::Array(items)) }) },
    Spec { name: "Slice", category: "List", inputs: &[l("list"), n("start", 0.0), n("end", -1.0)],
        outputs: &["list"],
        eval: Pure(|a, _| {
            let items = list(&a[0]);
            let len = items.len();
            let bound = |value: &Value, fallback: usize| match num(value) {
                Some(i) if i < 0.0 => (len as f64 + i + 1.0).max(0.0) as usize,
                Some(i) => (i as usize).min(len),
                None => fallback,
            };
            let (start, end) = (bound(&a[1], 0), bound(&a[2], len));
            out(Value::Array(items.get(start..end.max(start)).unwrap_or_default().to_vec()))
        }) },
    Spec { name: "Join Lists", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array([list(&a[0]), list(&a[1])].concat()))) },
    Spec { name: "Repeat Item", category: "List", inputs: &[l("item"), n("count", 3.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let count = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::Array(vec![a[0].clone(); count.min(MAX_ITEMS)]))
        }) },
    Spec { name: "Filter By Mask", category: "List", inputs: &[l("list"), l("mask")], outputs: &["in", "out"],
        eval: Pure(|a, _| {
            let (mut kept, mut dropped) = (Vec::new(), Vec::new());
            for (item, keep) in list(&a[0]).into_iter().zip(list(&a[1])) {
                if boolean(&keep).unwrap_or(false) { kept.push(item) } else { dropped.push(item) }
            }
            vec![Value::Array(kept), Value::Array(dropped)]
        }) },
    Spec { name: "Unique Items", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items: Vec<Value> = Vec::new();
            for item in list(&a[0]) {
                if !items.iter().any(|seen| equal(seen, &item)) { items.push(item) }
            }
            out(Value::Array(items))
        }) },
    Spec { name: "Index Of", category: "List", inputs: &[l("list"), l("item")], outputs: &["index"],
        eval: Pure(|a, _| out(number(list(&a[0]).iter().position(|x| equal(x, &a[1])).map_or(-1.0, |i| i as f64)))) },
    Spec { name: "Contains", category: "List", inputs: &[l("list"), l("item")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(list(&a[0]).iter().any(|x| equal(x, &a[1]))))) },
    Spec { name: "Chop", category: "List", inputs: &[l("list"), n("size", 2.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let size = num(&a[1]).unwrap_or(1.0).max(1.0) as usize;
            out(Value::Array(list(&a[0]).chunks(size).map(|chunk| Value::Array(chunk.to_vec())).collect()))
        }) },
    Spec { name: "Transpose", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| {
            let rows: Vec<Vec<Value>> = list(&a[0]).iter().map(list).collect();
            let width = rows.iter().map(Vec::len).min().unwrap_or(0);
            out(Value::Array((0..width).map(|k| Value::Array(rows.iter().map(|row| row[k].clone()).collect())).collect()))
        }) },
    Spec { name: "Empty List", category: "List", inputs: &[], outputs: &["list"],
        eval: Pure(|_, _| out(Value::Array(Vec::new()))) },
    Spec { name: "Is Empty", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(list(&a[0]).is_empty()))) },
    Spec { name: "Deconstruct", category: "List", inputs: LIST, outputs: &["first", "rest"],
        eval: Pure(|a, _| {
            let items = list(&a[0]);
            vec![items.first().cloned().unwrap_or(Value::Null), Value::Array(items.into_iter().skip(1).collect())]
        }) },
    Spec { name: "Rest Of Items", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(list(&a[0]).into_iter().skip(1).collect()))) },
    Spec { name: "Take Items", category: "List", inputs: &[l("list"), n("amount", 1.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            // A negative amount takes from the end.
            let items = list(&a[0]);
            let k = num(&a[1]).unwrap_or(0.0) as i64;
            let n = (k.unsigned_abs() as usize).min(items.len());
            out(Value::Array(if k < 0 { items[items.len() - n..].to_vec() } else { items[..n].to_vec() }))
        }) },
    Spec { name: "Drop Items", category: "List", inputs: &[l("list"), n("amount", 1.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            // A negative amount drops from the end.
            let items = list(&a[0]);
            let k = num(&a[1]).unwrap_or(0.0) as i64;
            let n = (k.unsigned_abs() as usize).min(items.len());
            out(Value::Array(if k < 0 { items[..items.len() - n].to_vec() } else { items[n..].to_vec() }))
        }) },
    Spec { name: "Add Item To Front", category: "List", inputs: &[l("list"), l("item")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array([vec![a[1].clone()], list(&a[0])].concat()))) },
    Spec { name: "Add Item To End", category: "List", inputs: &[l("list"), l("item")], outputs: &["list"],
        eval: Pure(|a, _| { let mut items = list(&a[0]); items.push(a[1].clone()); out(Value::Array(items)) }) },
    Spec { name: "Insert Item", category: "List", inputs: &[l("list"), l("item"), n("index", 0.0)],
        outputs: &["list"],
        eval: Pure(|a, _| {
            // A negative index counts from the end, -1 appending.
            let mut items = list(&a[0]);
            let len = items.len();
            let at = num(&a[2]).map_or(len, |i| {
                if i < 0.0 { (len as f64 + i + 1.0).max(0.0) as usize } else { (i as usize).min(len) }
            });
            items.insert(at, a[1].clone());
            out(Value::Array(items))
        }) },
    Spec { name: "Remove Item At Index", category: "List", inputs: &[l("list"), n("index", 0.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items = list(&a[0]);
            if let Some(i) = num(&a[1]).and_then(|i| index(items.len(), i)) { items.remove(i); }
            out(Value::Array(items))
        }) },
    Spec { name: "Replace Item At Index", category: "List", inputs: &[l("list"), n("index", 0.0), l("item")],
        outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items = list(&a[0]);
            if let Some(i) = num(&a[1]).and_then(|i| index(items.len(), i)) { items[i] = a[2].clone(); }
            out(Value::Array(items))
        }) },
    Spec { name: "Shift Indices", category: "List", inputs: &[l("list"), n("amount", 1.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items = list(&a[0]);
            if !items.is_empty() {
                let k = (num(&a[1]).unwrap_or(0.0) as i64).rem_euclid(items.len() as i64) as usize;
                items.rotate_right(k);
            }
            out(Value::Array(items))
        }) },
    Spec { name: "Cycle List", category: "List", inputs: &[l("list"), n("count", 2.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let items = list(&a[0]);
            let count = amount(&a[1]).min(MAX_ITEMS / items.len().max(1));
            out(Value::Array(items.iter().cloned().cycle().take(items.len() * count).collect()))
        }) },
    Spec { name: "Shuffle", category: "List", inputs: &[l("list"), n("seed", 1.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let mut items = list(&a[0]);
            let mut next = random(num(&a[1]).unwrap_or(1.0));
            for i in (1..items.len()).rev() {
                items.swap(i, ((next() * (i + 1) as f64) as usize).min(i));
            }
            out(Value::Array(items))
        }) },
    Spec { name: "Clean", category: "List", inputs: LIST, outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(list(&a[0]).into_iter().filter(|item| !item.is_null()).collect()))) },
    Spec { name: "All True", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(list(&a[0]).iter().all(|x| boolean(x) == Some(true))))) },
    Spec { name: "Any True", category: "List", inputs: LIST, outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(list(&a[0]).iter().any(|x| boolean(x) == Some(true))))) },
    Spec { name: "Set Union", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(set(&a[0], &a[1], true, |_| true)))) },
    Spec { name: "Set Intersection", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(set(&a[0], &a[1], false, |found| found)))) },
    Spec { name: "Set Difference", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: Pure(|a, _| out(Value::Array(set(&a[0], &a[1], false, |found| !found)))) },
    Spec { name: "Sort By Key", category: "List", inputs: &[l("list"), l("keys")], outputs: &["list", "keys"],
        eval: Pure(|a, _| {
            let (items, keys) = sort_by_keys(&a[0], &a[1]);
            vec![Value::Array(items), Value::Array(keys)]
        }) },
    Spec { name: "Sort Index By Value", category: "List", inputs: LIST, outputs: &["indices"],
        eval: Pure(|a, _| {
            let items = list(&a[0]);
            let mut order: Vec<usize> = (0..items.len()).collect();
            order.sort_by(|x, y| compare(&items[*x], &items[*y]).unwrap_or(Ordering::Equal));
            out(Value::Array(order.into_iter().map(|i| number(i as f64)).collect()))
        }) },
    Spec { name: "Group By Key", category: "List", inputs: &[l("list"), l("keys")], outputs: &["groups", "keys"],
        eval: Pure(|a, _| {
            let mut keys: Vec<Value> = Vec::new();
            let mut groups: Vec<Vec<Value>> = Vec::new();
            for (item, key) in list(&a[0]).into_iter().zip(list(&a[1])) {
                match keys.iter().position(|seen| equal(seen, &key)) {
                    Some(k) => groups[k].push(item),
                    None => {
                        keys.push(key);
                        groups.push(vec![item]);
                    }
                }
            }
            vec![Value::Array(groups.into_iter().map(Value::Array).collect()), Value::Array(keys)]
        }) },
    Spec { name: "Cartesian Product", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: Pure(|a, _| {
            let other = list(&a[1]);
            out(Value::Array(list(&a[0]).iter()
                .flat_map(|x| other.iter().map(move |y| Value::Array(vec![x.clone(), y.clone()])))
                .take(MAX_ITEMS)
                .collect()))
        }) },
    Spec { name: "Combinations", category: "List", inputs: &[l("list"), n("length", 2.0)], outputs: &["list"],
        eval: Pure(|a, _| {
            let mut found = Vec::new();
            combinations(&list(&a[0]), amount(&a[1]), 0, &mut Vec::new(), &mut found);
            out(Value::Array(found))
        }) },
    // ── Text ─────────────────────────────────────────────────────────────
    Spec { name: "Concatenate", category: "Text", inputs: &[s("a", ""), s("b", "")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(text(&a[0]) + &text(&a[1])))) },
    Spec { name: "Text Length", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| out(number(text(&a[0]).chars().count() as f64))) },
    Spec { name: "Upper Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(text(&a[0]).to_uppercase()))) },
    Spec { name: "Lower Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(text(&a[0]).to_lowercase()))) },
    Spec { name: "Split", category: "Text", inputs: &[s("text", ""), s("separator", ",")], outputs: &["list"],
        eval: Pure(|a, _| {
            let (source, separator) = (text(&a[0]), text(&a[1]));
            out(Value::Array(if separator.is_empty() {
                source.chars().map(|c| Value::String(c.to_string())).collect()
            } else {
                source.split(separator.as_str()).map(|part| Value::String(part.to_owned())).collect()
            }))
        }) },
    Spec { name: "Join Text", category: "Text", inputs: &[l("list"), s("separator", ",")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(list(&a[0]).iter().map(text).collect::<Vec<_>>().join(&text(&a[1]))))) },
    Spec { name: "Replace", category: "Text", inputs: &[s("text", ""), s("search", ""), s("replacement", "")],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let search = text(&a[1]);
            let source = text(&a[0]);
            out(Value::String(if search.is_empty() { source } else { source.replace(&search, &text(&a[2])) }))
        }) },
    Spec { name: "Substring", category: "Text", inputs: &[s("text", ""), n("start", 0.0), n("length", 1.0)],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let start = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            let length = num(&a[2]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::String(text(&a[0]).chars().skip(start).take(length).collect()))
        }) },
    Spec { name: "Text Contains", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(text(&a[0]).contains(&text(&a[1]))))) },
    Spec { name: "To Number", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| out(num(&a[0]).map_or(Value::Null, number))) },
    Spec { name: "To Text", category: "Text", inputs: &[l("value")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(text(&a[0])))) },
    Spec { name: "Starts With", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(text(&a[0]).starts_with(&text(&a[1]))))) },
    Spec { name: "Ends With", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: RESULT,
        eval: Pure(|a, _| out(Value::Bool(text(&a[0]).ends_with(&text(&a[1]))))) },
    Spec { name: "Text Index Of", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: &["index"],
        eval: Pure(|a, _| {
            let source = text(&a[0]);
            out(number(source.find(&text(&a[1])).map_or(-1.0, |byte| source[..byte].chars().count() as f64)))
        }) },
    Spec { name: "Trim", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| out(Value::String(text(&a[0]).trim().to_owned()))) },
    Spec { name: "Title Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: Pure(|a, _| {
            let mut start = true;
            out(Value::String(text(&a[0]).chars().map(|c| {
                let mapped: String = if start { c.to_uppercase().collect() } else { c.to_lowercase().collect() };
                start = c.is_whitespace();
                mapped
            }).collect()))
        }) },
    Spec { name: "Pad Left", category: "Text", inputs: &[s("text", ""), n("width", 8.0), s("pad", " ")],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let (source, pad) = (text(&a[0]), text(&a[2]).chars().next().unwrap_or(' '));
            let fill = amount(&a[1]).saturating_sub(source.chars().count());
            out(Value::String(std::iter::repeat_n(pad, fill).chain(source.chars()).collect()))
        }) },
    Spec { name: "Pad Right", category: "Text", inputs: &[s("text", ""), n("width", 8.0), s("pad", " ")],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let (source, pad) = (text(&a[0]), text(&a[2]).chars().next().unwrap_or(' '));
            let fill = amount(&a[1]).saturating_sub(source.chars().count());
            out(Value::String(source.chars().chain(std::iter::repeat_n(pad, fill)).collect()))
        }) },
    Spec { name: "Insert Text", category: "Text", inputs: &[s("text", ""), n("index", 0.0), s("insert", "")],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let chars: Vec<char> = text(&a[0]).chars().collect();
            let at = amount(&a[1]).min(chars.len());
            out(Value::String(
                chars[..at].iter().copied().chain(text(&a[2]).chars()).chain(chars[at..].iter().copied()).collect(),
            ))
        }) },
    Spec { name: "Remove Text", category: "Text", inputs: &[s("text", ""), n("start", 0.0), n("length", 1.0)],
        outputs: RESULT,
        eval: Pure(|a, _| {
            let (start, length) = (amount(&a[1]), amount(&a[2]));
            out(Value::String(text(&a[0]).chars().enumerate()
                .filter(|(i, _)| *i < start || *i >= start + length)
                .map(|(_, c)| c)
                .collect()))
        }) },
    // ── Geometry ─────────────────────────────────────────────────────────
    Spec { name: "Point", category: "Geometry", inputs: &[n("x", 0.0), n("y", 0.0), n("z", 0.0)], outputs: POINT,
        eval: Pure(|a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(y), Some(z)) => point_value([x, y, z]),
            _ => Value::Null,
        })) },
    Spec { name: "Point Components", category: "Geometry", inputs: &[v("point")], outputs: &["x", "y", "z"],
        eval: Pure(|a, _| match point(&a[0]) {
            Some(p) => p.map(number).to_vec(),
            None => vec![Value::Null; 3],
        }) },
    Spec { name: "Distance", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number(p.iter().zip(q).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()),
            _ => Value::Null,
        })) },
    Spec { name: "Midpoint", category: "Geometry", inputs: PQ, outputs: POINT,
        eval: Pure(|a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value([0, 1, 2].map(|i| (p[i] + q[i]) * 0.5)),
            _ => Value::Null,
        })) },
    Spec { name: "Interpolate", category: "Geometry", inputs: &[v("a"), v("b"), n("t", 0.5)], outputs: POINT,
        eval: Pure(|a, _| out(match (point(&a[0]), point(&a[1]), num(&a[2])) {
            (Some(p), Some(q), Some(t)) => point_value([0, 1, 2].map(|i| p[i] + (q[i] - p[i]) * t)),
            _ => Value::Null,
        })) },
    Spec { name: "Angle", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number((q[1] - p[1]).atan2(q[0] - p[0]).to_degrees()),
            _ => Value::Null,
        })) },
    Spec { name: "Translate", category: "Geometry",
        inputs: &[v("point"), n("dx", 0.0), n("dy", 0.0), n("dz", 0.0)], outputs: POINT,
        eval: Pure(|a, _| {
            let d = [1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            map_point(&a[0], |p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
        }) },
    Spec { name: "Polar Point", category: "Geometry", inputs: &[v("point"), n("angle", 0.0), n("distance", 1.0)],
        outputs: POINT,
        eval: Pure(|a, _| {
            let (angle, distance) = (num(&a[1]).unwrap_or(0.0), num(&a[2]).unwrap_or(0.0));
            let (sin, cos) = angle.to_radians().sin_cos();
            map_point(&a[0], |p| [p[0] + distance * cos, p[1] + distance * sin, p[2]])
        }) },
    Spec { name: "Rotate Point", category: "Geometry", inputs: &[v("point"), v("center"), n("angle", 90.0)],
        outputs: POINT,
        eval: Pure(|a, _| match point(&a[1]) {
            Some(center) => {
                let angle = num(&a[2]).unwrap_or(0.0);
                map_point(&a[0], |p| rotate_xy(p, center, angle))
            }
            None => out(Value::Null),
        }) },
    Spec { name: "Scale Point", category: "Geometry", inputs: &[v("point"), v("center"), n("factor", 2.0)],
        outputs: POINT,
        eval: Pure(|a, _| match point(&a[1]) {
            Some(c) => {
                let k = num(&a[2]).unwrap_or(1.0);
                map_point(&a[0], |p| [0, 1, 2].map(|i| c[i] + (p[i] - c[i]) * k))
            }
            None => out(Value::Null),
        }) },
    Spec { name: "Origin", category: "Geometry", inputs: &[], outputs: POINT,
        eval: Pure(|_, _| out(point_value([0.0; 3]))) },
    Spec { name: "Vector By Two Points", category: "Geometry", inputs: PQ, outputs: &["vector"],
        eval: Pure(|a, _| vectors(a, |p, q| point_value([0, 1, 2].map(|i| q[i] - p[i])))) },
    Spec { name: "Vector Length", category: "Geometry", inputs: &[v("vector")], outputs: RESULT,
        eval: Pure(|a, _| out(point(&a[0]).map_or(Value::Null, |p| number(length(p))))) },
    Spec { name: "Normalize", category: "Geometry", inputs: &[v("vector")], outputs: &["vector"],
        eval: Pure(|a, _| out(point(&a[0]).map_or(Value::Null, |p| {
            let k = length(p);
            if k == 0.0 { Value::Null } else { point_value(p.map(|x| x / k)) }
        }))) },
    Spec { name: "Add Vectors", category: "Geometry", inputs: PQ, outputs: &["vector"],
        eval: Pure(|a, _| vectors(a, |p, q| point_value([0, 1, 2].map(|i| p[i] + q[i])))) },
    Spec { name: "Subtract Vectors", category: "Geometry", inputs: PQ, outputs: &["vector"],
        eval: Pure(|a, _| vectors(a, |p, q| point_value([0, 1, 2].map(|i| p[i] - q[i])))) },
    Spec { name: "Scale Vector", category: "Geometry", inputs: &[v("vector"), n("factor", 2.0)], outputs: &["vector"],
        eval: Pure(|a, _| {
            let k = num(&a[1]).unwrap_or(1.0);
            map_point(&a[0], |p| p.map(|x| x * k))
        }) },
    Spec { name: "Dot Product", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| vectors(a, |p, q| number(dot(p, q)))) },
    Spec { name: "Cross Product", category: "Geometry", inputs: PQ, outputs: &["vector"],
        eval: Pure(|a, _| vectors(a, |p, q| point_value(cross(p, q)))) },
    Spec { name: "Angle Between", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: Pure(|a, _| vectors(a, |p, q| {
            let k = length(p) * length(q);
            if k == 0.0 { Value::Null } else { number((dot(p, q) / k).clamp(-1.0, 1.0).acos().to_degrees()) }
        })) },
];
