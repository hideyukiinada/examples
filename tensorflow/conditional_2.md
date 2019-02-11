# Applying operations conditionally based on a value in each sample in TensorFlow

When you use TensorFlow, you build a computational graph first, then call run() within a session
 to do the actual calculation unless you use it in the eager execution mode.
 
 This makes it coming up with code challenging to do what would be a simple task in Python.
 
 Let's take a look at the below example:

<img src="../assets/images/conditional_1.jpg" width="400px">

There are 4 samples in this dataset.
Each sample is a 3-dimensional vector.
This data is first defined as an ndarray, then fed to the tfSession.run() method in feed_dict as the value for a placeholder. 
 
```
     a = np.array([[0.8, 1, 100], [0, 1000, 5], [0.1, 2, 2000], [0, 2000, 4]])
```

The objective is the following:

* Calculate the sum of 2nd elements of each sample if the first element is greater than 0
* Calculate the sum of 3nd elements of each sample if the first element is 0

<img src="../assets/images/conditional_2.jpg"  width="600px">>

In this example, the first sample has 0.8 in the first element, therefore, keep the 2nd element and ignore the third element.
For the second element, the first element is 0, so ignore the second element and keep the third element.
Applying the same logic to the third and fourth elements, then you will get 3 for the first objective and 9 for the second objective.

As far as I know, you cannot use tf.cond method as it requires the condition to be a scalar number.

Here are the steps that I took to solve this problem.

## First objective
### Split data to three tensors
```
    x = tf.placeholder(tf.float32, shape=(None, 3), name="x")

    x_first_cell = x[:, 0]  # [0.8, 0., 0.1, 0.]
    x_second_cell = x[:, 1]  # [1.e+00, 1.e+03, 2.e+00, 2.e+03]
    x_third_cell = x[:, 2]  # [100., 5., 2000., 4.]
```

### Adjust the first tensor to be 1 if the value is above 0 by using the ceil() method
```
    x_first_cell_ceil = tf.ceil(x_first_cell)  # [1., 0., 1., 0.]

```

### Multiply this with the second tensor and calculate the sum
```
    x_filtered_second_cell = tf.multiply(x_first_cell_ceil, x_second_cell)  # [1., 0., 2., 0.]
    x_positive_sum = tf.reduce_sum(x_filtered_second_cell)  # 3.0
```

## Second objective
Second objective is slightly more cumbersome as you first negate the value of the first tensor.

### Negate first tensor
```
    x_first_cell_ceil_bool = tf.cast(x_first_cell_ceil, tf.bool)  # [True, False, True, False]
    x_first_cell_ceil_bool_negated_bool = tf.logical_not(x_first_cell_ceil_bool)  # [False, True, False, True]
    x_first_cell_ceil_negated = tf.cast(x_first_cell_ceil_bool_negated_bool, tf.float32)  # [0., 1., 0., 1.]
```

### # Multiply the third cell
```
    x_filtered_third_cell = tf.multiply(x_first_cell_ceil_negated, x_third_cell)  # [0., 5., 0., 4.]
    x_negative_sum = tf.reduce_sum(x_filtered_third_cell)  # 9.0
```

# Complete code
Here is the complete code of a function that does this:
```
def example():
    """An example code.
    """
    tf.reset_default_graph()

    # Features are not normalized for illustration purposes
    a = np.array([[0.8, 1, 100], [0, 1000, 5], [0.1, 2, 2000], [0, 2000, 4]])

    x = tf.placeholder(tf.float32, shape=(None, 3), name="x")

    x_first_cell = x[:, 0]  # [0.8, 0., 0.1, 0.]
    x_second_cell = x[:, 1]  # [1.e+00, 1.e+03, 2.e+00, 2.e+03]
    x_third_cell = x[:, 2]  # [100., 5., 2000., 4.]

    x_first_cell_ceil = tf.ceil(x_first_cell)  # [1., 0., 1., 0.]

    x_filtered_second_cell = tf.multiply(x_first_cell_ceil, x_second_cell)  # [1., 0., 2., 0.]
    x_positive_sum = tf.reduce_sum(x_filtered_second_cell)  # 3.0

    # Negate first cell
    x_first_cell_ceil_bool = tf.cast(x_first_cell_ceil, tf.bool)  # [True, False, True, False]
    x_first_cell_ceil_bool_negated_bool = tf.logical_not(x_first_cell_ceil_bool)  # [False, True, False, True]
    x_first_cell_ceil_negated = tf.cast(x_first_cell_ceil_bool_negated_bool, tf.float32)  # [0., 1., 0., 1.]

    # Multiply the third cell
    x_filtered_third_cell = tf.multiply(x_first_cell_ceil_negated, x_third_cell)  # [0., 5., 0., 4.]
    x_negative_sum = tf.reduce_sum(x_filtered_third_cell)  # 9.0

    init_op = tf.global_variables_initializer()  # Set up operator to assign all init values to variables

    with tf.Session() as s:
        s.run(init_op)  # Actually assign initial value to variables

        val = s.run(x_positive_sum, feed_dict={x: a})
        print("value of x: %s" % (str(val)))

        val = s.run(x_negative_sum, feed_dict={x: a})
        print("value of x: %s" % (str(val)))
```
