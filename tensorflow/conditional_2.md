# Applying operations conditionally based on a value in each sample in TensorFlow

When you use TensorFlow, you build a computational graph first, then call run() within a session
 to do the actual calculation unless you use it in the eager execution mode.
 
 This makes it coming up with code challenging to do what would be a simple task in Python.
 
 Let's take a look at the below example:

<img src="../assets/images/conditional_1.jpg" width="800px">


This data is fed to the run() method in feed_dict as the value for a placeholder. 
 
```
     a = np.array([[0.8, 1, 100], [0, 1000, 5], [0.1, 2, 2000], [0, 2000, 4]])
```

The objective is the following:

Calculate the sum of 2nd elements of each sample if the first element is greater than 0
Calculate the sum of 3nd elements of each sample if the first element is 0

As far as I know, you cannot use tf.cond method as it requires the condition to be a scalar number.

<img src="../assets/images/conditional_2.jpg"  width="800px">>


