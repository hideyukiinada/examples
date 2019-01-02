# Does truncation option matter in analyzing the sentiment in reviews?
By Hide Inada
<hr>
I did another analysis if truncating at the beginning of the review or the end of a review if the review
is longer than the size of the input for the network.  Here is the [script] used(https://github.com/hideyukiinada/examples/blob/master/3rdparty/keras/imdb_truncate_beginning_or_end.py).

### Accuracy
####  Truncate at the beginning
![Truncate at the end](/assets/images/imdb_t2.png)

####  Truncate at the end
![Truncate at the beginning](/assets/images/imdb_t4.png)

### Loss
####  Truncate at the beginning
![Truncate at the beginning](/assets/images/imdb_t1.png)

####  Truncate at the end
![Truncate at the end](/assets/images/imdb_t3.png)

# Summary
Based on this, I didn't see an obvious difference between these two options.
