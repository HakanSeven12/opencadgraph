//! The built-in node library.
//!
//! Each [`Spec`] names its inputs and outputs and evaluates one call. Inputs
//! not marked `list` are scalar: a list arriving there makes the node run once
//! per element (see [`crate::value::lace`]), so `Add` over two lists adds them
//! pairwise without any list-aware code in `Add` itself. Angles are degrees.

use std::cmp::Ordering;

use kernel::geom2d::{cross, Tolerance};
use serde_json::Value;

use crate::value::{boolean, list, num, number, point, point_value, text};
use crate::Host;

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

pub struct Spec {
    pub name: &'static str,
    pub category: &'static str,
    pub inputs: &'static [Input],
    pub outputs: &'static [&'static str],
    pub eval: fn(&[Value], &dyn Host) -> Vec<Value>,
}

impl std::fmt::Debug for Spec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

pub const CATEGORIES: &[&str] = &["Input", "Math", "Logic", "List", "Text", "Geometry"];

/// The spec called `name`.
pub fn spec(name: &str) -> Option<&'static Spec> {
    LIBRARY.iter().find(|spec| spec.name == name)
}

const fn n(name: &'static str, x: f64) -> Input {
    Input { name, default: Initial::Num(x), list: false }
}
const fn s(name: &'static str, value: &'static str) -> Input {
    Input { name, default: Initial::Text(value), list: false }
}
const fn b(name: &'static str, value: bool) -> Input {
    Input { name, default: Initial::Bool(value), list: false }
}
/// A scalar input with no default (points, curves, arbitrary items).
const fn v(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: false }
}
/// A whole-list input.
const fn l(name: &'static str) -> Input {
    Input { name, default: Initial::None, list: true }
}

const X: &[Input] = &[n("x", 0.0)];
const AB: &[Input] = &[n("a", 0.0), n("b", 0.0)];
const PQ: &[Input] = &[v("a"), v("b")];
const LIST: &[Input] = &[l("list")];
const TEXT: &[Input] = &[s("text", "")];
const CURVE: &[Input] = &[v("curve")];
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

fn same_plane(a: &kernel::space::Plane, b: &kernel::space::Plane) -> bool {
    let close = |p: [f64; 3], q: [f64; 3]| p.iter().zip(q).all(|(x, y)| (x - y).abs() <= 1e-9);
    close(a.origin, b.origin) && close(a.x_axis, b.x_axis) && close(a.y_axis, b.y_axis)
}

pub const LIBRARY: &[Spec] = &[
    // ── Input ────────────────────────────────────────────────────────────
    Spec { name: "Number", category: "Input", inputs: &[n("value", 0.0)], outputs: VALUE,
        eval: |a, _| out(num(&a[0]).map_or(Value::Null, number)) },
    Spec { name: "Number Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 0.1)], outputs: VALUE,
        eval: |a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        } },
    Spec { name: "Integer Slider", category: "Input",
        inputs: &[n("value", 5.0), n("min", 0.0), n("max", 10.0), n("step", 1.0)], outputs: VALUE,
        eval: |a, _| {
            let [value, min, max, step] = [0, 1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0).round());
            let snapped = if step > 0.0 { min + ((value - min) / step).round() * step } else { value };
            out(number(snapped.clamp(min.min(max), max.max(min))))
        } },
    Spec { name: "Text", category: "Input", inputs: &[s("value", "")], outputs: VALUE,
        eval: |a, _| out(Value::String(text(&a[0]))) },
    Spec { name: "Boolean", category: "Input", inputs: &[b("value", false)], outputs: VALUE,
        eval: |a, _| out(boolean(&a[0]).map_or(Value::Null, Value::Bool)) },
    Spec { name: "Formula", category: "Input",
        inputs: &[s("expression", "a+b"), n("a", 0.0), n("b", 0.0), n("c", 0.0), n("d", 0.0)],
        outputs: RESULT,
        eval: |a, host| out(host.expression(&substitute(&text(&a[0]), &a[1..])).map_or(Value::Null, number)) },
    Spec { name: "Watch", category: "Input", inputs: &[l("in")], outputs: &["out"],
        eval: |a, _| out(a[0].clone()) },
    // ── Math ─────────────────────────────────────────────────────────────
    Spec { name: "Add", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x + y) },
    Spec { name: "Subtract", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x - y) },
    Spec { name: "Multiply", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x * y) },
    Spec { name: "Divide", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x / y) },
    Spec { name: "Modulo", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, |x, y| x.rem_euclid(y)) },
    Spec { name: "Power", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::powf) },
    Spec { name: "Minimum", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::min) },
    Spec { name: "Maximum", category: "Math", inputs: AB, outputs: RESULT, eval: |a, _| f2(a, f64::max) },
    Spec { name: "Negate", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| -x) },
    Spec { name: "Absolute", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::abs) },
    Spec { name: "Square Root", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::sqrt) },
    Spec { name: "Exponential", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::exp) },
    Spec { name: "Natural Log", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::ln) },
    Spec { name: "Log10", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::log10) },
    Spec { name: "Sine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().sin()) },
    Spec { name: "Cosine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().cos()) },
    Spec { name: "Tangent", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.to_radians().tan()) },
    Spec { name: "Arcsine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.asin().to_degrees()) },
    Spec { name: "Arccosine", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.acos().to_degrees()) },
    Spec { name: "Arctangent", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, |x| x.atan().to_degrees()) },
    Spec { name: "Arctangent2", category: "Math", inputs: &[n("y", 0.0), n("x", 1.0)], outputs: RESULT,
        eval: |a, _| f2(a, |y, x| y.atan2(x).to_degrees()) },
    Spec { name: "Round", category: "Math", inputs: &[n("x", 0.0), n("digits", 0.0)], outputs: RESULT,
        eval: |a, _| f2(a, |x, d| { let m = 10f64.powi(d as i32); (x * m).round() / m }) },
    Spec { name: "Floor", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::floor) },
    Spec { name: "Ceiling", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::ceil) },
    Spec { name: "Clamp", category: "Math", inputs: &[n("x", 0.0), n("min", 0.0), n("max", 1.0)], outputs: RESULT,
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(lo), Some(hi)) => number(x.clamp(lo.min(hi), hi.max(lo))),
            _ => Value::Null,
        }) },
    Spec { name: "Remap", category: "Math",
        inputs: &[n("x", 0.0), n("from min", 0.0), n("from max", 1.0), n("to min", 0.0), n("to max", 1.0)],
        outputs: RESULT,
        eval: |a, _| {
            let values: Option<Vec<f64>> = a.iter().map(num).collect();
            out(values.map_or(Value::Null, |v| number(v[3] + (v[0] - v[1]) / (v[2] - v[1]) * (v[4] - v[3]))))
        } },
    Spec { name: "To Radians", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::to_radians) },
    Spec { name: "To Degrees", category: "Math", inputs: X, outputs: RESULT, eval: |a, _| f1(a, f64::to_degrees) },
    Spec { name: "Pi", category: "Math", inputs: &[], outputs: RESULT, eval: |_, _| out(number(std::f64::consts::PI)) },
    Spec { name: "E", category: "Math", inputs: &[], outputs: RESULT, eval: |_, _| out(number(std::f64::consts::E)) },
    Spec { name: "Random", category: "Math", inputs: &[n("seed", 1.0), n("count", 10.0)], outputs: &["list"],
        eval: |a, _| {
            let mut state = (num(&a[0]).unwrap_or(1.0) as u64) ^ 0x9E37_79B9_7F4A_7C15;
            let count = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::Array((0..count.min(MAX_ITEMS)).map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                number((state >> 11) as f64 / (1u64 << 53) as f64)
            }).collect()))
        } },
    // ── Logic ────────────────────────────────────────────────────────────
    Spec { name: "Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(Value::Bool(equal(&a[0], &a[1]))) },
    Spec { name: "Not Equal", category: "Logic", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(Value::Bool(!equal(&a[0], &a[1]))) },
    Spec { name: "Less", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_lt) },
    Spec { name: "Less Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_le) },
    Spec { name: "Greater", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_gt) },
    Spec { name: "Greater Or Equal", category: "Logic", inputs: PQ, outputs: RESULT, eval: |a, _| test(a, Ordering::is_ge) },
    Spec { name: "And", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x && y) },
    Spec { name: "Or", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x || y) },
    Spec { name: "Xor", category: "Logic", inputs: &[b("a", false), b("b", false)], outputs: RESULT,
        eval: |a, _| logic(a, |x, y| x != y) },
    Spec { name: "Not", category: "Logic", inputs: &[b("a", false)], outputs: RESULT,
        eval: |a, _| out(boolean(&a[0]).map_or(Value::Null, |x| Value::Bool(!x))) },
    Spec { name: "If", category: "Logic", inputs: &[b("test", true), v("then"), v("else")], outputs: RESULT,
        eval: |a, _| out(match boolean(&a[0]) {
            Some(true) => a[1].clone(),
            Some(false) => a[2].clone(),
            None => Value::Null,
        }) },
    // ── List ─────────────────────────────────────────────────────────────
    Spec { name: "Range", category: "List", inputs: &[n("start", 0.0), n("end", 10.0), n("step", 1.0)],
        outputs: &["list"],
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(end), Some(step)) if step != 0.0 && (end - start) / step >= 0.0 => {
                range(start, ((end - start) / step + 1e-9).floor() as usize + 1, step)
            }
            _ => Value::Array(Vec::new()),
        }) },
    Spec { name: "Sequence", category: "List", inputs: &[n("start", 0.0), n("count", 10.0), n("step", 1.0)],
        outputs: &["list"],
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(start), Some(count), Some(step)) => range(start, count.max(0.0) as usize, step),
            _ => Value::Array(Vec::new()),
        }) },
    Spec { name: "Create List", category: "List",
        inputs: &[l("item 0"), l("item 1"), l("item 2"), l("item 3"), l("item 4")], outputs: &["list"],
        eval: |a, _| out(Value::Array(a.iter().filter(|item| !item.is_null()).cloned().collect())) },
    Spec { name: "Count", category: "List", inputs: LIST, outputs: &["count"],
        eval: |a, _| out(number(list(&a[0]).len() as f64)) },
    Spec { name: "Get Item", category: "List", inputs: &[l("list"), n("index", 0.0)], outputs: &["item"],
        eval: |a, _| {
            let items = list(&a[0]);
            out(num(&a[1]).and_then(|i| index(items.len(), i)).map_or(Value::Null, |i| items[i].clone()))
        } },
    Spec { name: "First Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: |a, _| out(list(&a[0]).first().cloned().unwrap_or(Value::Null)) },
    Spec { name: "Last Item", category: "List", inputs: LIST, outputs: &["item"],
        eval: |a, _| out(list(&a[0]).last().cloned().unwrap_or(Value::Null)) },
    Spec { name: "Reverse", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| { let mut items = list(&a[0]); items.reverse(); out(Value::Array(items)) } },
    Spec { name: "Sort", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let mut items = list(&a[0]);
            items.sort_by(|x, y| compare(x, y).unwrap_or(Ordering::Equal));
            out(Value::Array(items))
        } },
    Spec { name: "Sum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(number(numbers(&a[0]).iter().sum())) },
    Spec { name: "Average", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| { let x = numbers(&a[0]); out(number(x.iter().sum::<f64>() / x.len() as f64)) } },
    Spec { name: "List Minimum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(numbers(&a[0]).into_iter().reduce(f64::min).map_or(Value::Null, number)) },
    Spec { name: "List Maximum", category: "List", inputs: LIST, outputs: RESULT,
        eval: |a, _| out(numbers(&a[0]).into_iter().reduce(f64::max).map_or(Value::Null, number)) },
    Spec { name: "Flatten", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| { let mut items = Vec::new(); flatten(&a[0], &mut items); out(Value::Array(items)) } },
    Spec { name: "Slice", category: "List", inputs: &[l("list"), n("start", 0.0), n("end", -1.0)],
        outputs: &["list"],
        eval: |a, _| {
            let items = list(&a[0]);
            let len = items.len();
            let bound = |value: &Value, fallback: usize| match num(value) {
                Some(i) if i < 0.0 => (len as f64 + i + 1.0).max(0.0) as usize,
                Some(i) => (i as usize).min(len),
                None => fallback,
            };
            let (start, end) = (bound(&a[1], 0), bound(&a[2], len));
            out(Value::Array(items.get(start..end.max(start)).unwrap_or_default().to_vec()))
        } },
    Spec { name: "Join Lists", category: "List", inputs: &[l("a"), l("b")], outputs: &["list"],
        eval: |a, _| out(Value::Array([list(&a[0]), list(&a[1])].concat())) },
    Spec { name: "Repeat Item", category: "List", inputs: &[l("item"), n("count", 3.0)], outputs: &["list"],
        eval: |a, _| {
            let count = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::Array(vec![a[0].clone(); count.min(MAX_ITEMS)]))
        } },
    Spec { name: "Filter By Mask", category: "List", inputs: &[l("list"), l("mask")], outputs: &["in", "out"],
        eval: |a, _| {
            let (mut kept, mut dropped) = (Vec::new(), Vec::new());
            for (item, keep) in list(&a[0]).into_iter().zip(list(&a[1])) {
                if boolean(&keep).unwrap_or(false) { kept.push(item) } else { dropped.push(item) }
            }
            vec![Value::Array(kept), Value::Array(dropped)]
        } },
    Spec { name: "Unique Items", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let mut items: Vec<Value> = Vec::new();
            for item in list(&a[0]) {
                if !items.iter().any(|seen| equal(seen, &item)) { items.push(item) }
            }
            out(Value::Array(items))
        } },
    Spec { name: "Index Of", category: "List", inputs: &[l("list"), l("item")], outputs: &["index"],
        eval: |a, _| out(number(list(&a[0]).iter().position(|x| equal(x, &a[1])).map_or(-1.0, |i| i as f64))) },
    Spec { name: "Contains", category: "List", inputs: &[l("list"), l("item")], outputs: RESULT,
        eval: |a, _| out(Value::Bool(list(&a[0]).iter().any(|x| equal(x, &a[1])))) },
    Spec { name: "Chop", category: "List", inputs: &[l("list"), n("size", 2.0)], outputs: &["list"],
        eval: |a, _| {
            let size = num(&a[1]).unwrap_or(1.0).max(1.0) as usize;
            out(Value::Array(list(&a[0]).chunks(size).map(|chunk| Value::Array(chunk.to_vec())).collect()))
        } },
    Spec { name: "Transpose", category: "List", inputs: LIST, outputs: &["list"],
        eval: |a, _| {
            let rows: Vec<Vec<Value>> = list(&a[0]).iter().map(list).collect();
            let width = rows.iter().map(Vec::len).min().unwrap_or(0);
            out(Value::Array((0..width).map(|k| Value::Array(rows.iter().map(|row| row[k].clone()).collect())).collect()))
        } },
    // ── Text ─────────────────────────────────────────────────────────────
    Spec { name: "Concatenate", category: "Text", inputs: &[s("a", ""), s("b", "")], outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]) + &text(&a[1]))) },
    Spec { name: "Text Length", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(number(text(&a[0]).chars().count() as f64)) },
    Spec { name: "Upper Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]).to_uppercase())) },
    Spec { name: "Lower Case", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]).to_lowercase())) },
    Spec { name: "Split", category: "Text", inputs: &[s("text", ""), s("separator", ",")], outputs: &["list"],
        eval: |a, _| {
            let (source, separator) = (text(&a[0]), text(&a[1]));
            out(Value::Array(if separator.is_empty() {
                source.chars().map(|c| Value::String(c.to_string())).collect()
            } else {
                source.split(separator.as_str()).map(|part| Value::String(part.to_owned())).collect()
            }))
        } },
    Spec { name: "Join Text", category: "Text", inputs: &[l("list"), s("separator", ",")], outputs: RESULT,
        eval: |a, _| out(Value::String(list(&a[0]).iter().map(text).collect::<Vec<_>>().join(&text(&a[1])))) },
    Spec { name: "Replace", category: "Text", inputs: &[s("text", ""), s("search", ""), s("replacement", "")],
        outputs: RESULT,
        eval: |a, _| {
            let search = text(&a[1]);
            let source = text(&a[0]);
            out(Value::String(if search.is_empty() { source } else { source.replace(&search, &text(&a[2])) }))
        } },
    Spec { name: "Substring", category: "Text", inputs: &[s("text", ""), n("start", 0.0), n("length", 1.0)],
        outputs: RESULT,
        eval: |a, _| {
            let start = num(&a[1]).unwrap_or(0.0).max(0.0) as usize;
            let length = num(&a[2]).unwrap_or(0.0).max(0.0) as usize;
            out(Value::String(text(&a[0]).chars().skip(start).take(length).collect()))
        } },
    Spec { name: "Text Contains", category: "Text", inputs: &[s("text", ""), s("search", "")], outputs: RESULT,
        eval: |a, _| out(Value::Bool(text(&a[0]).contains(&text(&a[1])))) },
    Spec { name: "To Number", category: "Text", inputs: TEXT, outputs: RESULT,
        eval: |a, _| out(num(&a[0]).map_or(Value::Null, number)) },
    Spec { name: "To Text", category: "Text", inputs: &[l("value")], outputs: RESULT,
        eval: |a, _| out(Value::String(text(&a[0]))) },
    // ── Geometry ─────────────────────────────────────────────────────────
    Spec { name: "Point", category: "Geometry", inputs: &[n("x", 0.0), n("y", 0.0), n("z", 0.0)], outputs: POINT,
        eval: |a, _| out(match (num(&a[0]), num(&a[1]), num(&a[2])) {
            (Some(x), Some(y), Some(z)) => point_value([x, y, z]),
            _ => Value::Null,
        }) },
    Spec { name: "Point Components", category: "Geometry", inputs: &[v("point")], outputs: &["x", "y", "z"],
        eval: |a, _| match point(&a[0]) {
            Some(p) => p.map(number).to_vec(),
            None => vec![Value::Null; 3],
        } },
    Spec { name: "Distance", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number(p.iter().zip(q).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()),
            _ => Value::Null,
        }) },
    Spec { name: "Midpoint", category: "Geometry", inputs: PQ, outputs: POINT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => point_value([0, 1, 2].map(|i| (p[i] + q[i]) * 0.5)),
            _ => Value::Null,
        }) },
    Spec { name: "Interpolate", category: "Geometry", inputs: &[v("a"), v("b"), n("t", 0.5)], outputs: POINT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1]), num(&a[2])) {
            (Some(p), Some(q), Some(t)) => point_value([0, 1, 2].map(|i| p[i] + (q[i] - p[i]) * t)),
            _ => Value::Null,
        }) },
    Spec { name: "Angle", category: "Geometry", inputs: PQ, outputs: RESULT,
        eval: |a, _| out(match (point(&a[0]), point(&a[1])) {
            (Some(p), Some(q)) => number((q[1] - p[1]).atan2(q[0] - p[0]).to_degrees()),
            _ => Value::Null,
        }) },
    Spec { name: "Translate", category: "Geometry",
        inputs: &[v("point"), n("dx", 0.0), n("dy", 0.0), n("dz", 0.0)], outputs: POINT,
        eval: |a, _| {
            let d = [1, 2, 3].map(|i| num(&a[i]).unwrap_or(0.0));
            map_point(&a[0], |p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
        } },
    Spec { name: "Polar Point", category: "Geometry", inputs: &[v("point"), n("angle", 0.0), n("distance", 1.0)],
        outputs: POINT,
        eval: |a, _| {
            let (angle, distance) = (num(&a[1]).unwrap_or(0.0), num(&a[2]).unwrap_or(0.0));
            let (sin, cos) = angle.to_radians().sin_cos();
            map_point(&a[0], |p| [p[0] + distance * cos, p[1] + distance * sin, p[2]])
        } },
    Spec { name: "Rotate Point", category: "Geometry", inputs: &[v("point"), v("center"), n("angle", 90.0)],
        outputs: POINT,
        eval: |a, _| match point(&a[1]) {
            Some(center) => {
                let angle = num(&a[2]).unwrap_or(0.0);
                map_point(&a[0], |p| rotate_xy(p, center, angle))
            }
            None => out(Value::Null),
        } },
    Spec { name: "Scale Point", category: "Geometry", inputs: &[v("point"), v("center"), n("factor", 2.0)],
        outputs: POINT,
        eval: |a, _| match point(&a[1]) {
            Some(c) => {
                let k = num(&a[2]).unwrap_or(1.0);
                map_point(&a[0], |p| [0, 1, 2].map(|i| c[i] + (p[i] - c[i]) * k))
            }
            None => out(Value::Null),
        } },
    Spec { name: "Curve Length", category: "Geometry", inputs: CURVE, outputs: RESULT,
        eval: |a, host| out(host.curve(&a[0]).map_or(Value::Null, |c| number(c.length()))) },
    Spec { name: "Curve Area", category: "Geometry", inputs: CURVE, outputs: RESULT,
        eval: |a, host| out(host.curve(&a[0]).map_or(Value::Null, |c| number(c.curve.enclosed_area().abs()))) },
    Spec { name: "Curve Endpoints", category: "Geometry", inputs: CURVE, outputs: &["start", "end"],
        eval: |a, host| match host.curve(&a[0]) {
            Some(c) => vec![point_value(c.point_at(0.0)), point_value(c.point_at(1.0))],
            None => vec![Value::Null; 2],
        } },
    Spec { name: "Point At Parameter", category: "Geometry", inputs: &[v("curve"), n("t", 0.5)], outputs: POINT,
        eval: |a, host| out(match (host.curve(&a[0]), num(&a[1])) {
            (Some(c), Some(t)) => point_value(c.point_at(t)),
            _ => Value::Null,
        }) },
    Spec { name: "Point At Distance", category: "Geometry", inputs: &[v("curve"), n("distance", 0.0)],
        outputs: POINT,
        eval: |a, host| out(match (host.curve(&a[0]), num(&a[1])) {
            (Some(c), Some(d)) => point_value(c.point_at_distance(d)),
            _ => Value::Null,
        }) },
    Spec { name: "Parameter At Point", category: "Geometry", inputs: &[v("curve"), v("point")], outputs: &["t"],
        eval: |a, host| out(match (host.curve(&a[0]), point(&a[1])) {
            (Some(c), Some(p)) => c.parameter_at(p).map_or(Value::Null, number),
            _ => Value::Null,
        }) },
    Spec { name: "Divide Curve", category: "Geometry", inputs: &[v("curve"), n("count", 10.0)], outputs: &["points"],
        eval: |a, host| out(match (host.curve(&a[0]), num(&a[1])) {
            (Some(c), Some(count)) if count >= 1.0 => {
                let (count, length) = ((count as usize).min(MAX_ITEMS), c.length());
                Value::Array((0..=count).map(|k| point_value(c.point_at_distance(length * k as f64 / count as f64))).collect())
            }
            _ => Value::Null,
        }) },
    Spec { name: "Intersect", category: "Geometry", inputs: &[v("a"), v("b")], outputs: &["points"],
        eval: |a, host| out(match (host.curve(&a[0]), host.curve(&a[1])) {
            // Curves on different planes meet at most in isolated points the
            // plane-curve intersection cannot see; they report none.
            (Some(p), Some(q)) if same_plane(&p.plane, &q.plane) => Value::Array(
                cross::intersect(&p.curve, &q.curve, Tolerance::new(1e-9))
                    .into_iter()
                    .map(|crossing| point_value(p.plane.point_at(crossing.point)))
                    .collect(),
            ),
            _ => Value::Array(Vec::new()),
        }) },
];
