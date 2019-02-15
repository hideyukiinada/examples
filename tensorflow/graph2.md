## Overview

In TensorFlow, you will build a computational graph with variables and operations first, then let TensorFlow do the actual calculation within a session.

In this article, I will discuss how variables and operations are actually turned into nodes on a graph.

## Code Example

Below is a short code snippet that has a few very simple operations.  The main logic is:
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

If you run the above code, you will see the animals.pbtxt generated.  The content of the file is rather long, so let's see the list of nodes.
Here are the list of node with its name:

* jungle/lion/Initializer/Const
* jungle/lion
* jungle/lion/Assign
* jungle/lion/read
* jungle/add/y
* jungle/add
* jungle/Assign

As you can see, all the names are prefixed with the word "jungle".
This is because of the following line:

```
    with tf.variable_scope("jungle") as scope:
```

For example, if we had declared the variable scope as:

```
    with tf.variable_scope("forest") as scope:
```

All the names would have been prefixed with "forest".

### First group of nodes
Now let's look at the nodes below:
* jungle/lion/Initializer/Const
* jungle/lion
* jungle/lion/Assign

jungle/lion/Assign has input keys:
```
node {
  name: "jungle/lion/Assign"
  op: "Assign"
  input: "jungle/lion"
  input: "jungle/lion/Initializer/Const"
```

Based on this, you can tell that the below line resulted in a initializer constant node and a variable node:
```
        lion_ref = tf.get_variable("lion", [], dtype=tf.float32, initializer=tf.constant_initializer(123))
```

### Second group of nodes
Next one is a little more complex:

* jungle/lion/read
* jungle/add/y
* jungle/add
* jungle/Assign

Obviously, all of these are coming from this line:
```
        assign_lion = tf.assign(lion_ref, lion_ref + 27)
```

Let's do one by one.
Logical operations that are happening on this line are:

1) Create a constant node
2) Read the value of the lion variable node
3) Add these two nodes
4) Assign the sum to the lion variable node

Let's have a look at the "jungle/add" node:
```
node {
  name: "jungle/add"
  op: "Add"
  input: "jungle/lion/read"
  input: "jungle/add/y"
```
This corresponds to the third item "Add these two nodes", and 

```
node {
  name: "jungle/lion/read"
  op: "Identity"
  input: "jungle/lion"
```
will tell that "jungle/lion/read" maps to second item "Read the value of the lion variable node".

As for "jungle/add/y", the value is set to 27 so it is for the first item "Create a constant node".
```
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
```

Now, the remaining item is the fourth item "Assign the sum to the lion variable node", which "jungle/Assign" node was created for:
```
node {
  name: "jungle/Assign"
  op: "Assign"
  input: "jungle/lion"
  input: "jungle/add"
```

## Final words
Hopefully, this article gave you some ideas about how each variable and operations in the source code get turned into actual nodes on a graph.

TensorFlow's website also has a good article [A Tool Developer's Guide to TensorFlow Model Files] (https://www.tensorflow.org/guide/extend/model_files) to cover the high-level concepts in this area, and I recommend it.

<hr>

# Generated pbtxt file
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
