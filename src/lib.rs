//! A node graph engine.
//!
//! Nodes are operations ([`Spec`]s: the built-in [`ops::LIBRARY`] plus any the
//! application defines) or host objects whose property rows are the ports.
//! The engine owns the graph and the dataflow — ordering, list lacing, value
//! coercion — and asks the [`Host`] for everything application-specific. It
//! has no user interface, no document model and no dependency on either; with
//! the unit host `()` it evaluates the built-in library on its own.

pub mod expr;
pub mod ops;
pub mod value;

use std::collections::BTreeMap;

use serde_json::{json, Value};

pub use ops::{spec, Eval, Spec, CATEGORIES, LIBRARY};

pub type NodeId = u32;

/// What the graph needs from the application. Every method has a default, so
/// an application implements only what it offers.
pub trait Host {
    /// Makes node `node` own `instances.len()` objects of `kind`, writes each
    /// instance's fields, and returns every instance's readable fields.
    fn objects(
        &mut self,
        _node: NodeId,
        _kind: &str,
        _instances: &[Vec<(String, Value)>],
    ) -> Vec<Vec<(String, Value)>> {
        Vec::new()
    }

    /// Evaluates one call of an application node ([`Eval::Host`]) on node
    /// `node`, returning its outputs in `spec.outputs` order.
    fn call(&mut self, _node: NodeId, _spec: &'static Spec, _args: &[Value]) -> Vec<Value> {
        Vec::new()
    }

    /// Evaluates an arithmetic expression.
    fn expression(&self, text: &str) -> Option<f64> {
        expr::eval(text)
    }
}

/// No application: the built-in library only.
impl Host for () {}

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

    /// The graph as JSON: each node's kind and parameters, and the links.
    /// Outputs are not stored; they come back from the next evaluation.
    pub fn to_json(&self) -> Value {
        let port = |port: &Port| json!({ "node": port.node, "field": port.field });
        json!({
            "nodes": self.nodes.iter().map(|node| json!({
                "id": node.id,
                "kind": match &node.kind {
                    Kind::Op(spec) => json!({ "op": spec.name }),
                    Kind::Object(name) => json!({ "object": name }),
                },
                "params": node.params,
            })).collect::<Vec<_>>(),
            "links": self.links.iter().map(|link| json!({
                "from": port(&link.from),
                "to": port(&link.to),
            })).collect::<Vec<_>>(),
        })
    }

    /// Reads [`to_json`](Self::to_json) output, finding operations by name
    /// through `resolve` (e.g. the application's specs, then [`spec`]).
    /// Inputs an operation has gained since the file was written start at
    /// their defaults; links to nodes that are not there are dropped.
    pub fn from_json(
        value: &Value,
        resolve: &dyn Fn(&str) -> Option<&'static Spec>,
    ) -> Result<Self, String> {
        let mut graph = Graph::default();
        for node in value["nodes"].as_array().ok_or("missing nodes")? {
            let id = node["id"]
                .as_u64()
                .and_then(|id| NodeId::try_from(id).ok())
                .filter(|id| *id > 0 && graph.node(*id).is_none())
                .ok_or("missing, zero or repeated node id")?;
            let kind = if let Some(name) = node["kind"]["op"].as_str() {
                Kind::Op(resolve(name).ok_or_else(|| format!("unknown node `{name}`"))?)
            } else if let Some(name) = node["kind"]["object"].as_str() {
                Kind::Object(name.to_owned())
            } else {
                return Err(format!("node {id} has no kind"));
            };
            graph.next_id = id - 1;
            graph.add(kind);
            if let (Some(node), Some(params)) = (graph.nodes.last_mut(), node["params"].as_object()) {
                node.params.extend(params.iter().map(|(name, value)| (name.clone(), value.clone())));
            }
            graph.next_id = graph.nodes.iter().map(|node| node.id).max().unwrap_or(0);
        }
        let port = |value: &Value| -> Option<Port> {
            Some(Port {
                node: NodeId::try_from(value["node"].as_u64()?).ok()?,
                field: value["field"].as_str()?.to_owned(),
            })
        };
        for link in value["links"].as_array().map(Vec::as_slice).unwrap_or_default() {
            if let (Some(from), Some(to)) = (port(&link["from"]), port(&link["to"])) {
                if graph.node(from.node).is_some() && graph.node(to.node).is_some() {
                    graph.connect(from, to);
                }
            }
        }
        Ok(graph)
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

    fn evaluate_op(&self, id: NodeId, spec: &'static Spec, host: &mut dyn Host) -> BTreeMap<String, Value> {
        let args: Vec<Value> = spec
            .inputs
            .iter()
            .map(|input| self.input(&Port { node: id, field: input.name.to_owned() }))
            .collect();
        let scalar: Vec<bool> = spec.inputs.iter().map(|input| !input.list).collect();
        let values = value::lace(&args, &scalar, spec.outputs.len(), &mut |args| match spec.eval {
            Eval::Pure(eval) => eval(args, &*host),
            Eval::Host => host.call(id, spec, args),
        });
        spec.outputs
            .iter()
            .map(|name| (*name).to_owned())
            .zip(values)
            .collect()
    }

    fn evaluate_object(&self, id: NodeId, kind: &str, host: &mut dyn Host) -> BTreeMap<String, Value> {
        // Linked fields, then the values set by hand on unlinked ones.
        let mut linked: Vec<(String, Value)> = self
            .links
            .iter()
            .filter(|link| link.to.node == id)
            .map(|link| (link.to.field.clone(), self.input(&link.to)))
            .collect();
        if let Some(node) = self.node(id) {
            linked.extend(
                node.params
                    .iter()
                    .filter(|(field, _)| !self.is_linked(&Port { node: id, field: (*field).clone() }))
                    .map(|(field, value)| (field.clone(), value.clone())),
            );
        }
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
