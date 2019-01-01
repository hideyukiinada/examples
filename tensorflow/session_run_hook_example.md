# How to use tf.train.SessionRunHook
by Hide Inada

I was looking at CIFAR10 example code and saw that they are subclassing tf.train.SessionRunHook to create a
custom class called  _LoggerHook class to log events related to sessions.
There is a [Stack Overflow page](https://stackoverflow.com/questions/45532365/is-there-any-tutorial-for-tf-train-sessionrunhook) where you can get the general idea, but I wanted to have a crispier understanding regarding how it works, so I created an example doc and here is the output. 

## Code
```

class MySessionRunHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        print("MySessionRunHook.after_create_session() called")

    def begin(self):
        print("MySessionRunHook.begin() called")

    def end(self, session):
        print("MySessionRunHook.end() called")

    def before_run(self, run_context):
        print("MySessionRunHook.before_run() called")

    def after_run(self, run_context, run_values):
        print("MySessionRunHook.after_run() called")


def example():
    """An example code.
    """
    tf.reset_default_graph()

    with tf.variable_scope("vs1") as scope:
        a = tf.get_variable("a", [2, 3], dtype=tf.float32, initializer=tf.constant_initializer(50))
        b = tf.get_variable("b", [2, 3], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(seed=0))

        c = tf.placeholder(tf.float32, [2, 3])
        sum_op1 = tf.add(a, b)
        sum_op = tf.add(sum_op1, c)

    init_op = tf.global_variables_initializer()  # Set up operator to assign all init values to variables

    print("Before 'with tf.train.MonitoredTrainingSession'")

    with tf.train.MonitoredTrainingSession(hooks=[MySessionRunHook()]) as s:
        print("Before s.run(init_op)")
        s.run(init_op)  # Actually assign initial value to variables
        print("After s.run(init_op)")

        print("Before sum = s.run #1 call")
        sum = s.run(sum_op, feed_dict={c: [[100, 200, 300], [400, 500, 600]]})
        print("After sum = s.run #1 call")
        print(sum)

        print("Before sum = s.run #2 call")
        sum = s.run(sum_op, feed_dict={c: [[-10, -20, -30], [-40, -50, -60]]})
        print("After sum = s.run #2 call")
        print(sum)


```

## Output
```
Before 'with tf.train.MonitoredTrainingSession'
MySessionRunHook.begin() called
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Graph was finalized.
2018-12-31 18:10:53.203224: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Done running local_init_op.
MySessionRunHook.after_create_session() called
Before s.run(init_op)
MySessionRunHook.before_run() called
MySessionRunHook.after_run() called
After s.run(init_op)
Before sum = s.run #1 call
MySessionRunHook.before_run() called
MySessionRunHook.after_run() called
After sum = s.run #1 call
[[149.1084  249.23465 348.97034]
 [450.27014 549.137   650.40607]]
Before sum = s.run #2 call
MySessionRunHook.before_run() called
MySessionRunHook.after_run() called
After sum = s.run #2 call
[[39.108402  29.234653  18.970352 ]
 [10.270153  -0.8629837 -9.593918 ]]
MySessionRunHook.end() called
```
