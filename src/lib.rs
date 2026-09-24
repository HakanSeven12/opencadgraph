//! A node graph engine for CAD.
//!
//! Nodes are either library operations ([`ops::LIBRARY`]) or host objects:
//! drawing objects whose property rows are the ports. The engine owns the
//! graph and the dataflow — ordering, list lacing, value coercion — and asks
//! the [`Host`] to create, edit and read objects. It has no user interface
//! and no document model of its own.

pub mod ops;
pub mod value;

use std::collections::BTreeMap;

use kernel::space::PlanarCurve;
use serde_json::Value;

pub use ops::{spec, Spec, CATEGORIES, LIBRARY};

pub type NodeId = u32;

/// What the graph needs from the application.
pub trait Host {
    /// Makes node `node` own `instances.len()` objects of `kind`, writes each
    /// instance's linked fields, and returns every instance's readable fields.
    fn objects(
        &mut self,
        node: NodeId,
        kind: &str,
        instances: &[Vec<(String, Value)>],
    ) -> Vec<Vec<(String, Value)>>;

    /// The curve an object value stands for.
    fn curve(&self, object: &Value) -> Option<PlanarCurve>;

    /// Evaluates an arithmetic expression.
    fn expression(&self, text: &str) -> Option<f64>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Port {
    pub node: NodeId,
    pub field: String,
}

#[derive(Clone, Debug)]
pub struct Link {
    pub from: Port,
    pub to: Port,
}

#[derive(Clone, Debug)]
pub enum Kind {
    Op(&'static Spec),
    /// A host object type, e.g. `"Line"`.
    Object(String),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub kind: Kind,
    /// Values of unlinked op inputs.
    pub params: BTreeMap<String, Value>,
    /// Results of the last evaluation.
    pub outputs: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    next_id: NodeId,
}

impl Graph {
    pub fn add(&mut self, kind: Kind) -> NodeId {
        self.next_id += 1;
        let params = match &kind {
            Kind::Op(spec) => spec
                .inputs
                .iter()
                .map(|input| (input.name.to_owned(), input.default.value()))
                .collect(),
            Kind::Object(_) => BTreeMap::new(),
        };
        self.nodes.push(Node {
            id: self.next_id,
            kind,
            params,
            outputs: BTreeMap::new(),
        });
        self.next_id
    }

    /// Removes a node and every link touching it.
    pub fn remove(&mut self, id: NodeId) -> Option<Node> {
        self.links.retain(|link| link.from.node != id && link.to.node != id);
        let index = self.nodes.iter().position(|node| node.id == id)?;
        Some(self.nodes.remove(index))
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|node| node.id == id)
    }

    /// Links an output to an input, replacing the input's previous link.
    /// Refuses a link that would close a loop.
    pub fn connect(&mut self, from: Port, to: Port) -> bool {
        if from.node == to.node || self.reaches(to.node, from.node) {
            return false;
        }
        self.links.retain(|link| link.to != to);
        self.links.push(Link { from, to });
        true
    }

    /// Removes and returns the link into `to`.
    pub fn disconnect(&mut self, to: &Port) -> Option<Link> {
        let index = self.links.iter().position(|link| &link.to == to)?;
        Some(self.links.remove(index))
    }

    pub fn is_linked(&self, to: &Port) -> bool {
        self.links.iter().any(|link| &link.to == to)
    }

    /// True when data flows from `from` to `to`.
    pub fn reaches(&self, from: NodeId, to: NodeId) -> bool {
        let mut stack = vec![from];
        let mut seen = Vec::new();
        while let Some(id) = stack.pop() {
            if id == to {
                return true;
            }
            if !seen.contains(&id) {
                seen.push(id);
                stack.extend(
                    self.links
                        .iter()
                        .filter(|link| link.from.node == id)
                        .map(|link| link.to.node),
                );
            }
        }
        false
    }

    /// Nodes ordered so that every link's source precedes its target.
    pub fn order(&self) -> Vec<NodeId> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut pending: Vec<NodeId> = self.nodes.iter().map(|node| node.id).collect();
        while !pending.is_empty() {
            let ready: Vec<NodeId> = pending
                .iter()
                .copied()
                .filter(|id| {
                    !self
                        .links
                        .iter()
                        .any(|link| link.to.node == *id && pending.contains(&link.from.node))
                })
                .collect();
            if ready.is_empty() {
                break;
            }
            pending.retain(|id| !ready.contains(id));
            order.extend(ready);
        }
        order
    }

    /// The value arriving at an input: its link's source output, else the
    /// node's own parameter.
    pub fn input(&self, port: &Port) -> Value {
        if let Some(link) = self.links.iter().find(|link| &link.to == port) {
            return self
                .node(link.from.node)
                .and_then(|node| node.outputs.get(&link.from.field))
                .cloned()
                .unwrap_or(Value::Null);
        }
        self.node(port.node)
            .and_then(|node| node.params.get(&port.field))
            .cloned()
            .unwrap_or(Value::Null)
    }

    /// Evaluates every node, upstream first.
    pub fn evaluate(&mut self, host: &mut dyn Host) {
        for id in self.order() {
            let Some(kind) = self.node(id).map(|node| node.kind.clone()) else {
                continue;
            };
            let outputs = match kind {
                Kind::Op(spec) => self.evaluate_op(id, spec, host),
                Kind::Object(kind) => self.evaluate_object(id, &kind, host),
            };
            if let Some(node) = self.node_mut(id) {
                node.outputs = outputs;
            }
        }
    }

    fn evaluate_op(&self, id: NodeId, spec: &'static Spec, host: &dyn Host) -> BTreeMap<String, Value> {
        let args: Vec<Value> = spec
            .inputs
            .iter()
            .map(|input| self.input(&Port { node: id, field: input.name.to_owned() }))
            .collect();
        let scalar: Vec<bool> = spec.inputs.iter().map(|input| !input.list).collect();
        let values = value::lace(&args, &scalar, spec.outputs.len(), &mut |args| {
            (spec.eval)(args, host)
        });
        spec.outputs
            .iter()
            .map(|name| (*name).to_owned())
            .zip(values)
            .collect()
    }

    fn evaluate_object(&self, id: NodeId, kind: &str, host: &mut dyn Host) -> BTreeMap<String, Value> {
        let linked: Vec<(String, Value)> = self
            .links
            .iter()
            .filter(|link| link.to.node == id)
            .map(|link| (link.to.field.clone(), self.input(&link.to)))
            .collect();
        // One object per element of the shortest linked list. ponytail: an
        // empty list still keeps one object, since its ports are read from it.
        let count = linked
            .iter()
            .filter_map(|(_, value)| value.as_array().map(Vec::len))
            .min()
            .unwrap_or(1)
            .max(1);
        let instances: Vec<Vec<(String, Value)>> = (0..count)
            .map(|k| {
                linked
                    .iter()
                    .filter_map(|(field, value)| match value {
                        Value::Array(items) => items.get(k).map(|item| (field.clone(), item.clone())),
                        other => Some((field.clone(), other.clone())),
                    })
                    .collect()
            })
            .collect();
        let results = host.objects(id, kind, &instances);
        if results.len() == 1 {
            return results.into_iter().next().unwrap_or_default().into_iter().collect();
        }
        let mut outputs: BTreeMap<String, Value> = BTreeMap::new();
        for instance in results {
            for (field, value) in instance {
                if let Value::Array(items) =
                    outputs.entry(field).or_insert_with(|| Value::Array(Vec::new()))
                {
                    items.push(value);
                }
            }
        }
        outputs
    }
}
