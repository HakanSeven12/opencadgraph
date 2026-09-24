//! Port values.
//!
//! A value is plain JSON: numbers, text, booleans, `null`, lists (arrays) and
//! points (`{"x", "y", "z"}` objects). Object ports carry whatever the host
//! puts there. Every node reads its inputs through the lenient coercions
//! below, so a property shown as text (`"10.0000"`) feeds a number input.

use serde_json::{json, Value};

/// A number, from a number, a numeric string or a boolean.
pub fn num(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// A boolean, from a boolean, a number (non-zero) or `true`/`yes`/`1` text.
pub fn boolean(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|x| x != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Display text: strings as they are, `null` empty, anything else as JSON.
pub fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// A number value; NaN and infinities become `null`.
pub fn number(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// A point, from `{"x", "y"[, "z"]}`.
pub fn point(value: &Value) -> Option<[f64; 3]> {
    let object = value.as_object()?;
    let axis = |key: &str| object.get(key).and_then(num);
    Some([axis("x")?, axis("y")?, axis("z").unwrap_or(0.0)])
}

pub fn point_value(p: [f64; 3]) -> Value {
    json!({ "x": number(p[0]), "y": number(p[1]), "z": number(p[2]) })
}

/// The items of a list; a single value is a one-item list, `null` none.
pub fn list(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    }
}

/// Runs `f` once per element when a `scalar` input receives a list.
///
/// Lists on scalar inputs are walked in step, stopping at the shortest; other
/// inputs are passed whole to every call. Nested lists recurse, so a list of
/// lists maps element by element too. Returns `outputs` values, each a list
/// when lacing happened.
pub fn lace(
    inputs: &[Value],
    scalar: &[bool],
    outputs: usize,
    f: &mut dyn FnMut(&[Value]) -> Vec<Value>,
) -> Vec<Value> {
    let shortest = inputs
        .iter()
        .zip(scalar)
        .filter_map(|(value, scalar)| match value {
            Value::Array(items) if *scalar => Some(items.len()),
            _ => None,
        })
        .min();
    let Some(len) = shortest else {
        let mut result = f(inputs);
        result.resize(outputs, Value::Null);
        return result;
    };
    let mut columns = vec![Vec::with_capacity(len); outputs];
    for k in 0..len {
        let row: Vec<Value> = inputs
            .iter()
            .zip(scalar)
            .map(|(value, scalar)| match value {
                Value::Array(items) if *scalar => items[k].clone(),
                other => other.clone(),
            })
            .collect();
        for (column, value) in columns.iter_mut().zip(lace(&row, scalar, outputs, f)) {
            column.push(value);
        }
    }
    columns.into_iter().map(Value::Array).collect()
}
