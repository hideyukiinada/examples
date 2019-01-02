# Does word sequence matter in analyzing the sentiment in sentences?

Does word sequence matter in analyzing the sentiment in sentences?  

My answer was "Of course, it would."

However, I was reading [Keras team's IMDB sentiment analysis tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification) and played with its [companion code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb) with and without keeping the sequence of words, I'm not sure if I was right.

The example code squishes the word embedding of all the words and calculates the average for the entire sentence before feeding the data to a dense layer.
Another approach is to keep the word embedding and feed into the neural network as Mr. Jason Brownlee is doing in his example code(https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/).

So I tweaked the example code to also support the second case to see which option yields more accurate result.
Here is the result.

### Accuracy
![With Average Pooling](/assets/images/imdb2.png)
![Without Average Pooling](/assets/images/imdb4.png)

### Loss
![With Average Pooling](/assets/images/imdb1.png)
![Without Average Pooling](/assets/images/imdb3.png)

As you can see, 
