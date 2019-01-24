# What does TensorFlow's tf.control_dependencies do?
by Hide Inada

Have you seen tf.control_dependencies used in code written with TensorFlow?
It has a fancy name, so it may not be clear what it does but it's actually very straightforward, 
so I decided to write some example code to demo.

In case you are interested, here is [the API reference](https://www.tensorflow.org/api_docs/python/tf/control_dependencies) 

Here is my code:
```
 with tf.variable_scope("home_actitivies") as scope:
        cups_of_coffee = tf.get_variable("cups_of_coffee", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(0))

        drink_coffee = tf.assign(cups_of_coffee, cups_of_coffee + 1)

    with tf.variable_scope("work_actitivies") as scope:
        number_of_emails_read = tf.get_variable("number_of_emails_read", (), dtype=tf.int32,
                                                initializer=tf.constant_initializer(0))

        read_email = tf.assign(number_of_emails_read, number_of_emails_read + 1)
```

The above code defines:
* a variable to keep track of how many cups of coffee you had
* an operator to increase the number of cups you had
* a variable to keep track of how many emails you read
* an operator to increase the number of emails you read

You can specify each variable & operator in Session.run() in a piece of code like below:
```
 with tf.Session() as s:
        s.run(init_op)

        run_result = s.run(number_of_emails_read)
        print("1. Number of emails you read:%d" % (run_result))

        run_result = s.run(cups_of_coffee)
        print("   Number of cups of coffee you had:%d" % (run_result))

        print("2. Calling s.run(read_email)")
        run_result = s.run(read_email)

        run_result = s.run(number_of_emails_read)
        print("3. Check how many emails you read:%d" % (run_result))

        run_result = s.run(cups_of_coffee)
        print("   Number of cups of coffee you had:%d" % (run_result))
```

Running the code will produce the below output:

```
1. Number of emails you read:0
   Number of cups of coffee you had:0
2. Calling s.run(read_email)
3. Check how many emails you read:1
   Number of cups of coffee you had:0
```
As you can see, though you have read 1 email, the number of cups of coffee has not increased and stays at 0. This is a problem caused by the fact that the code defines *no relationship between coffee and email*.
Specifically, read_email operator is not connected to either the drink_coffee operator or the cups_of_coffee variable.
As it's important that you are fully awake before you read email so that you don't misunderstand
work email content, you might want to make sure that you drink coffee before you read email.

tf.control_dependencies makes this possible ;-)
By enclosing the read_email statement in the tf.control_dependencies([drink_coffee]) block, you can make sure that the drink_coffee operator is called before the read_email operator is called.

Here is the updated code:
```
    with tf.variable_scope("home_actitivies") as scope:
        cups_of_coffee = tf.get_variable("cups_of_coffee", (), dtype=tf.int32,
                                         initializer=tf.constant_initializer(0))

        drink_coffee = tf.assign(cups_of_coffee, cups_of_coffee + 1)


    with tf.variable_scope("work_actitivies") as scope:
        number_of_emails_read = tf.get_variable("number_of_emails_read", (), dtype=tf.int32,
                                                initializer=tf.constant_initializer(0))

        with tf.control_dependencies([drink_coffee]):
            read_email = tf.assign(number_of_emails_read, number_of_emails_read + 1)

```

Output is now
```
1. Number of emails you read:0
   Number of cups of coffee you had:0
2. Calling s.run(read_email)
3. Check how many emails you read:1
   Number of cups of coffee you had:1
```
Now, the cups of coffee also incremented.

[Complete code](control_dependencies) is in this repo.
