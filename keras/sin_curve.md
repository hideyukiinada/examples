# Limitation of Neural Networks

One of the advantages of neural networks is ability to fit a non-linear curve.

For example, I generated a sine curve that is shifted upward by np.random.random() * 0.3.
The below code can fit the curve well in -3PI to 3PI.  
```
    x = (np.random.random(1024) * 2.0 - 1.0)  # Continuous uniform distribution [-1, 1)
    x *= 3 * np.pi  # [-3pi, 3pi) # three cycles

    y = np.sin(x) + np.random.random() * 0.3

    x_test = (np.random.random(256) * 2.0 - 1.0)
    x_test *= 3 * np.pi

    y_test = np.sin(x_test)

    input_layer = Input(shape=(1,))
    layer = Dense(2048, activation='relu')(input_layer)
    layer = Dense(128, activation='relu')(layer)
    last_layer = Dense(1, activation='linear')(layer)

    model = Model(input_layer, last_layer)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(adam, loss=keras.losses.mean_squared_error)
    model.fit(x=x, y=y, epochs=100, validation_data=(x_test, y_test))

    plt.plot(x, y, 'bo')

    y_test = model.predict(x_test)
    plt.plot(x_test, y_test, 'rx')

    plt.show()
```
![Chart generated](https://github.com/hideyukiinada/examples/blob/master/assets/images/sine1.png)

As long as the test data is close to the training data, it can do a decent prediction.

## Unseen test data
What do you think will happen if test data is something the network has not seen?
I expanded the domain of x by changing:
```
    x_test *= 3 * np.pi
```
to
```
    x_test *= 6 * np.pi
```

Here is the result:
![Chart generated](https://github.com/hideyukiinada/examples/blob/master/assets/images/sine2.png)



