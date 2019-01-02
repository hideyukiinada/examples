# Does word sequence matter in analyzing the sentiment in sentences?
By Hide Inada
<hr>
Sentiment analysis in AI is used for assessing if a person had a positive sentiment or negative sentiment about a thing that the person is providing a review for.
Do you think the sequence of words in a review matter during this assessment, or it does not?

My answer before today was "Of course, it would."

However, I was reading [Keras team's IMDB sentiment analysis tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification) and played with its [companion code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb) with and without keeping the sequence of words, I'm not sure if I was right.

The example code squishes the word embedding of words in a sentence and calculates the average for the entire sentence before feeding the data to a dense layer.
Another approach is to keep the word embedding and feed into the neural network.

So I tweaked the example code to also support the second case to see which option yields more accurate result.
Here is [the example code with my tweaks](https://github.com/hideyukiinada/examples/blob/master/3rdparty/keras/imdb.py).

Let's have a look at the result. 
Please note that "Average Pooling:True" means that the word sequence was *not* kept.  "Average Pooling: False" means that the word sequence was kept.  What average pooling does is to take the embedding (vector) assigned to each word and take an average for the entire sentence.  In Keras, you use GlobalAveragePooling1D layer for this, and I wrote [an example script] (https://github.com/hideyukiinada/examples/blob/master/keras/average_pooling_1d_example) for this.

### Accuracy
####  Word sequence was *not* kept
![Word sequence not kept (With Average Pooling)](/assets/images/imdb2.png)

####  Word sequence was kept
![Word sequence kept (Without Average Pooling)](/assets/images/imdb4.png)

As you can see, accuracy goes up much faster for the option where the word sequence is kept without average pooling.
However, accuracy with average pooling eventually goes up to high 80's as well.  So this seems to be a tie.

### Loss
####  Word sequence was *not* kept
![Word sequence not kept (With Average Pooling)](/assets/images/imdb1.png)

####  Word sequence was kept
![Word sequence kept (Without Average Pooling)](/assets/images/imdb3.png)

For the chart without average pooling, you can see the its overfitting the training data as the validation loss splits around 5 epochs and the gap between the validation loss and the training data loss gets wider and wider. For this particular exercise, the increased loss did not seem to affect the accuracy, but it could in other cases.

# Recommended approach
First assess if your objective is impacted by the word sequence.  For example, if your goal is to translate a language into another language, the word sequence matters. For example, a sentence "John ate a fish" means different from "a fish ate John" ;-)  In this case, you have to keep the word sequence.

However, if you are not sure if the word sequence matters as the sentiment analysis in this case, conduct analysis with and without ignoring the word sequence.
As shown in this case, ignoring word sequence can potentially allow your network to handle diverse ways of each word appearing in a sentence that is not in the training dataset, thus making the prediction consistent.




