# Does truncation option matter in analyzing the sentiment in sentences?
By Hide Inada
<hr>
I did another analysis if truncating at the beginning of the review or the end of the review if the review
is longer than the size of the input for the network.

### Accuracy
####  Word sequence was *not* kept
![Truncate at the end](/assets/images/imdb_t2.png)

####  Word sequence was kept
![Truncate at the beginning](/assets/images/imdb_t4.png)

### Loss
####  Word sequence was *not* kept
![Truncate at the beginnin](/assets/images/imdb_t1.png)

####  Word sequence was kept
![Truncate at the end](assets/images/imdb_t3.png)

# Summary
Based on this, I didn't see an obvious difference between these two options.
