# opencadgraph

A node graph engine for CAD applications, with no dependencies beyond
`serde_json`.

Nodes are operations or host objects — application objects whose property
rows are the node's ports. The engine owns ordering, list lacing and value
coercion, and ships a basic library (input, math, logic, list, text, point
geometry) that runs on its own: `graph.evaluate(&mut ())`.

An application implements [`Host`](src/lib.rs) for what is its own: object
nodes (`Host::objects`) and further operations, declared as `Spec`s with
`Eval::Host` and evaluated by `Host::call`. It also chooses the node types it
offers and resolves them by name when reading a saved graph
(`Graph::from_json(value, &resolve)`).

Values are JSON: numbers, text, booleans, lists, and points as
`{"x", "y", "z"}`. A list reaching a scalar input runs the node once per
element; a list reaching an object property makes one object per element.

License: MPL-2.0
