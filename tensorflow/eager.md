# Eager Execution of TensorFlow - Huge help in playing with an API

Are you one of the developers who wished that TensorFlow programming could be done without too much boilerplate code?

When I first saw a TensorFlow example code, I saw something like below just to do an addition of two variables:

```
    with tf.variable_scope("addition") as scope:
        a = tf.get_variable("a", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(1))

        b = tf.get_variable("b", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(2))

        c = tf.add(a, b)

    init_op = tf.global_variables_initializer()

    with tf.Session() as s:
        s.run(init_op)

        c_value = s.run(c)
        print("Value of c: %d" % (c_value))

```

You get the following result:
```
Value of c: 3
```

I understood that this is because TensorFlow adopted an architecture to build a computation graph.
However, this makes it very difficult to verify API's functionality in Python console.
On my Mac Terminal, Python console is always open so that I can do a quick calculation or test an API.

Later on in October 2017, [TensorFlow introduced a feature that addresses this issue](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html).
It's called [Eager Execution](https://www.tensorflow.org/api_docs/python/tf/enable_eager_execution).

With eager execution, the previous code can be written as:
```
    tf.enable_eager_execution()

    a = tf.get_variable("a", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(1))

    b = tf.get_variable("b", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(2))

    c = tf.add(a, b)
    print("Value of c: %d" % (c))
```

You get the same result:
```
Value of c: 3
```

The whole code is now short enough that you can type into Python console.  For the sake of simplicity, I changed tf.get_variable() to tf.Variable() and typed in Python console to get below:

```
>>> import tensorflow as tf
>>> tf.enable_eager_execution()
>>> a = tf.Variable(1, "a")
>>> b = tf.Variable(2, "b")
>>> c = tf.add(a, b)
>>> c
<tf.Tensor: id=19, shape=(), dtype=int32, numpy=3>
>>> print("Value of c: %d" % (c))
Value of c: 3
```

You can also try an array:
```
>>> a = tf.Variable([[1, 2], [3, 4]])
>>> b = tf.Variable([[10, 20], [30, 40]])
>>> c = tf.add(a, b)
>>> print("Value of c: %s" % (str(c)))
Value of c: tf.Tensor(
[[11 22]
 [33 44]], shape=(2, 2), dtype=int32)
>>> d = tf.reshape(c, (1, 4))
>>> print("Value of d: %s" % (str(d)))
Value of d: tf.Tensor([[11 22 33 44]], shape=(1, 4), dtype=int32)
```

I hope this tip will help you play with TensorFlow a little bit more easily.
