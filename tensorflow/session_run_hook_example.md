I was looking at CIFAR10 example code and saw that they are subclassing tf.train.SessionRunHook to create a
custom class called  _LoggerHook class to log events related to sessions.
I wasn't really sure how it works, so I created an example doc and here is the output. 

Output
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
