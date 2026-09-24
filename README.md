# opencadgraph

A node graph engine for CAD applications.

Nodes are library operations (input, math, logic, list, text, geometry) or
host objects — drawing objects whose property rows are the node's ports. The
engine owns ordering, list lacing and value coercion; the application
implements [`Host`](src/lib.rs) to create, edit and read its objects and to
supply curves for the geometry nodes (built on
[opencadkernel](https://github.com/HakanSeven12/opencadkernel)).

Values are JSON: numbers, text, booleans, lists, and points as
`{"x", "y", "z"}`. A list reaching a scalar input runs the node once per
element; a list reaching an object property makes one object per element.

License: MPL-2.0
