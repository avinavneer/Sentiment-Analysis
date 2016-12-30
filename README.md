# Sentiment-Analysis
## Text mining and classification to predict the polarity of movie reviews  

Objective- Analyze movie revies and allot a positive or negative polarity sentiment. The data had been extracted by Pang, Lee and Vaidyanathan (2004) as part of their research into sentiment analysis. 
The data can be found at: https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz 

The movie reviews have been labeled as positive or negative by the original researchers and a supervised learning approach was followed for this project. The data was divided into training and test sets and different classification models were developed and their accuracies compared. As a baseline model, an average sentiment score of the review was calculated on the basis of the individual sentiment scores of the words, based on a unigram approach. The net positive or negative sentiment was then calculated to predict the final sentiment. The AFINN scores were used for scoring individual words- http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010. These accuracy of this baseline model was also compared to the more advanced models.


 
