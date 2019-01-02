# Does word sequence matter in analyzing the sentiment in sentences?
By Hide Inada
<hr>
Sentiment analysis in AI is used for assessing if a person had a positive sentiment or negative sentiment about a thing that the person is providing reviews for.

The question is do you think the sequence of words in a review matter during this assessment, or it does not.

My answer before today was "Of course, it would."

However, I was reading [Keras team's IMDB sentiment analysis tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification) and played with its [companion code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb) with and without keeping the sequence of words, I'm not sure if I was right.

The example code squishes the word embedding of all the words and calculates the average for the entire sentence before feeding the data to a dense layer.
Another approach is to keep the word embedding and feed into the neural network.

So I tweaked the example code to also support the second case to see which option yields more accurate result.
Here is the result.

### Accuracy
![With Average Pooling](/assets/images/imdb2.png)
![Without Average Pooling](/assets/images/imdb4.png)

As you can see, accuracy goes up much faster for the option where the word sequence is kept without average pooling.
However, accuracy with average pooling eventually goes up to high 80's as well.  So this seems to be a tie.

### Loss
![With Average Pooling](/assets/images/imdb1.png)
![Without Average Pooling](/assets/images/imdb3.png)

For the chart without average pooling, you can see the its overfitting the training data as the validation loss splits around 5 epochs and the gap between the validation loss and the training data loss gets wider and wider. For this exercise, the increased loss did not seem to affect the accuracy, but it could in other cases.

Here is my hypothesis.

If I were to write a review by using the word "fantastic", it could be:

* "I thought the movie was fantastic.".
* "It was the one of the most fantastic movies of the year!"
* "Fantastic cast and directing, well done!"

In a case like this, it doesn't really matter where the word "fantastic" is in a sentence as long as its gramatically correct.

Therefore, if you train the model with "was fantastic" then you may not catch "most fantastic" or "fantastic cast".

So what I would take is do both and analyze both results.

Of course, this is totally different from a language translation where the word sequence totally matters where John ate a fish mean different from a fish ate John ;-)


