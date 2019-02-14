In TensorFlow, you will build a computational graph with variables and operations first, then let TensorFlow do the actual calculation within a session.

In this article, I will discuss how each variable and operations are actually turned into a node on a graph.

In the below code, the main logic is:
* Declare one variable with the initial value 123
* Increment the variable by 27

```
def example():
    """An example code.
    """

    graph_dir = Path(GRAPH_DIR)
    if graph_dir.exists() is False:
        graph_dir.mkdir(parents=True, exist_ok=True)

    with tf.variable_scope("jungle") as scope:
        lion_ref = tf.get_variable("lion", [], dtype=tf.float32, initializer=tf.constant_initializer(123))
        assign_lion = tf.assign(lion_ref, lion_ref + 27)

    default_graph = tf.get_default_graph()
    graph_def = default_graph.as_graph_def()

    tf.io.write_graph(graph_def, GRAPH_DIR, "animals.pbtxt")
```

A graph is stored in the Graph object, so you need to convert to a GraphDef object to write to the file system.  Once you convert, you can call tf.io.write_graph to create a text file.
You may have noticed that pbtxt extension of the file, and a GraphDef object is using protobuf serialization to store graph data.

Here is the generated file:

```
node {
  name: "jungle/lion/Initializer/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@jungle/lion"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 123.0
      }
    }
  }
}
node {
  name: "jungle/lion"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@jungle/lion"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "jungle/lion/Assign"
  op: "Assign"
  input: "jungle/lion"
  input: "jungle/lion/Initializer/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@jungle/lion"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "jungle/lion/read"
  op: "Identity"
  input: "jungle/lion"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@jungle/lion"
      }
    }
  }
}
node {
  name: "jungle/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 27.0
      }
    }
  }
}
node {
  name: "jungle/add"
  op: "Add"
  input: "jungle/lion/read"
  input: "jungle/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "jungle/Assign"
  op: "Assign"
  input: "jungle/lion"
  input: "jungle/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@jungle/lion"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
versions {
  producer: 27
}
```

There is a lot of information, so let's see the list of nodes.
Here are the list of node with its name:

* jungle/lion/Initializer/Const
* jungle/lion
* jungle/lion/Assign
* jungle/lion/read
* jungle/add/y
* jungle/add
* jungle/Assign

