
##Natural Language Processing- sentiment analysis##

# removing the variables from the evironment
rm(list=ls())

# installing the required packages

install.packages('tm')
install.packages('dplyr')
install.packages('qdap')
install.packages('topicmodels')
install.packages('ggplot2')
install.packages('knitr')
install.packages('wordcloud')
install.packages("e1071")

# installing the RTextTools package for maximum entropy, boosting, glmnet etc.
install.packages("RTextTools")

install.packages("SnowballC")
install.packages("stringr")

# installing the caret package for getting the confusion matrix, training/validation data splitting etc.
install.packages("caret",dependencies = TRUE)

# installing the package for decision tree modeling
install.packages("rpart")

# installing the package for xgboost modeling
install.packages("xgboost")

# Loading the above installed packages one by one
library(caret)
library(rpart)
library(tm)
library(qdap)
library(ggplot2)
library(wordcloud)
library(knitr)
library(e1071)
library(RTextTools)
library(SnowballC)
library(stringr)
library(xgboost)
library(Matrix)

# doing the garbage collection for reslving the memory leakage issues 
gc()

# User need to change the directory location here.
negative_dir_location = "C://Users/Test/Desktop/review_polarity/txt_sentoken/neg"
positive_dir_location = "C://Users/Test/Desktop/review_polarity/txt_sentoken/pos"

# setting the Negative and Postive Corpus directory source
negative = Corpus(DirSource(negative_dir_location))   
positive = Corpus(DirSource(negative_dir_location))

# assigning the negative and positive corpus to variable doc
docs<- c(negative,positive)

# WriteLine will allow to acces the docs variable. For example sccess the indexes like 555, 1555 etc
writeLines(as.character(docs[[555]]))

# Get the different functions available
getTransformations()                                                              

# remove the punctuations from the corpus
docs = tm_map(docs, removePunctuation)

# remove the corpus to lower case
docs = tm_map(docs, tolower)                                                      
writeLines(as.character(docs[[555]]))

# remove the numbers from the corpus
docs = tm_map(docs, removeNumbers)                                                
writeLines(as.character(docs[[987]]))

# remove the Words using the stopword "Smart" available in tm package
docs <- tm_map(docs, removeWords,stopwords("Smart"))

# remove the white spaces from the corpus
docs <- tm_map(docs, stripWhitespace)

# Apply the stemming.
docs <- tm_map(docs,stemDocument)

# Converting the docs corpus to a plain text doument
docs<- tm_map(docs,PlainTextDocument)

# Create the DTM[Doument Term Matrix] based on the docs corpus
dtm <- DocumentTermMatrix(docs, control = list(weighting=weightTfIdf,
                                          minWorldLength=3,minDocFreq=2000))
dtm

# Converting dtm into a data frame
matrix<- as.data.frame(inspect(dtm))
ncol(matrix)

# Create response column verdict to the data frame matrix
matrix$verdict = as.factor(c(rep(0,1000),rep(1,1000)))

# printing out the summary of verdict column in the matrix data frame 
summary(factor(matrix$verdict))

############################################### 
## Baseline model using a sentiment score list
###############################################

# User need to set the affin list folder location
afinn_list_dir = "C://Users/Test/Desktop/imm6010/AFINN/AFINN-111.txt"

# Reading the polarity list
afinn_list <- read.delim(file=afinn_list_dir, header=FALSE, stringsAsFactors=FALSE)

# assigning the column names as word and score 
names(afinn_list) <- c('word', 'score')

# Converting the word column to lower case
afinn_list$word <- tolower(afinn_list$word)

#Categorizing words as positive and negative 
NegativeTerms <- afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4| afinn_list$score==-3 | afinn_list$score==-2 |afinn_list$score==-1]
PositiveTerms <- afinn_list$word[afinn_list$score==5 | afinn_list$score==4 | afinn_list$score==3|afinn_list$score==2|afinn_list$score==1]

# Creating the vector terms with column names of document term matrix[dtm]
terms = colnames(dtm)

# Grepping the Negative terms in terms
negative_terms = terms[terms%in% NegativeTerms]

# Grepping the Positive terms in terms
positive_terms = terms[terms%in% PositiveTerms]

# creating the negative scores based on the row sum of negative_terms columns
negative_scores = rowSums(as.matrix(dtm[,negative_terms]))

# creating the positive scores based on the row sum of positive_terms columns
positive_scores = rowSums(as.matrix(dtm[,positive_terms]))

# Differencing teh positive and negative scores
score = positive_scores-negative_scores
score

# creating the sentiment scores and 
matrix$sent_score=ifelse(score<=0,0,1)

# Cross tabing the sen_score and verdict columns
table(matrix$sent_score,matrix$verdict)

## Visualisation:

# Generate word clouds (positive and negative).
wordcloud(positive_terms,colSums(as.matrix(dtm[ , positive_terms])),
          min.freq=1,
          scale=c(4,0.75),colors=brewer.pal(n=9,"Blues")[5:10])

wordcloud(negative_terms,colSums(as.matrix(dtm[ , negative_terms])),
          min.freq=1,
          scale=c(4,0.7),
          color=brewer.pal(n=9, "Reds")[5:10])

# Reducing the sparsity of the dtm to 99%
dtm4 = removeSparseTerms(dtm,0.99)
dtm4

# Converting to matrix a bigger dtm
matrix2<- as.data.frame(inspect(dtm4))
ncol(matrix2)

# Create the verdict response columns by filling 1 against positive and 0 against negative
matrix2$verdict = as.factor(c(rep(0,1000),rep(1,1000)))
summary(factor(matrix2$verdict))

# Creating a stratified training and validation set using caret package
partition <- createDataPartition(y = matrix2$verdict, p = 0.6, list = FALSE)
train <- matrix2[partition,]
val <- matrix2[-partition,]

############################# 
# Decision Tree
############################# 

# Training the model
tree_model <- rpart(verdict~.,data=train,method ="class")

# Predicting the validation set based on the model
predict_tree = predict(tree_model,newdata=val,type="class")

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confusionMatrix1 = confusionMatrix(predict_tree,val$verdict)
confusionMatrix1

# Reducing the dimenstionality and training again
tree_model2 <- rpart(verdict ~ action+back+bad+character+good+great+worth+perfect+performance+
                 love+scene+make+people+story+time+worst+special+nice+interest+fun+humor+
                 classic+boring+extremely+entertaining+evil+experience+hell, 
               data = train, method = "class")

# Applying the dimenstionality model into the validation data set
predict_tree2 <- predict(tree_model2, newdata = val, type = "class")

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confustionMatrix <- confusionMatrix(predict_tree2, val$verdict)
confustionMatrix

#####################################################################
# Using the RTextTools package
# Use the create container concept for different modeling techniques 
#####################################################################

# Creating a container to be passed for modeling 
container<- create_container(dtm, matrix$verdict[1:2000], 
                             trainSize = c(1:500,1001:1500),
                             testSize = c(501:1000,1501:2000),
                             virgin = FALSE)

# SVM Using RTextTools
svm_rtt<-train_model(container,"SVM")
svm_classify<- classify_model(container,svm_rtt)
create_analytics(container,svm_classify)

# GLMNET Using RTextTools
glmnet<- train_model(container,"GLMNET")
glmnet_classify<- classify_model(container,glmnet)
create_analytics(container,glmnet_classify)

# Maximum Entropy Using RTextTools
maxent<- train_model(container,"MAXENT")
maxent_classify<- classify_model(container,maxent)
create_analytics(container,maxent_classify)

# Boosting Using RTextTools
boosting<- train_model(container,'BOOSTING')
boosting_classify<- classify_model(container,boosting)
create_analytics(container,boosting_classify)

# Summarizing the results obtained
create_precisionRecallSummary(container,cbind(glmnet_classify,svm_classify,maxent_classify,
                                              boosting_classify))


############################ 
# Naive Bayesian Modeling
############################ 

# Training the model
NaiveBayes1 = naiveBayes(verdict~.,data=train)

# Predicting the validation set based on the model
predict_nb1 = predict(NaiveBayes1,newdata=val)

# Creating the Confustion Matrix(caret package) for evaluating the matrixes
confusionMatrixnb1 = confusionMatrix(predict_nb1,val$verdict)
confusionMatrixnb1

# Reducing the dimenstionality and training again
NaiveBayes2 = naiveBayes(verdict~ action+back+bad+character+good+great+worth+perfect+performance+
                             love+scene+make+people+story+time+worst+special+nice+interest+fun+humor+
                             classic+boring+extremely+entertaining+evil+experience+hell, 
                           data = train)

# Applying the dimenstionality model into the validation data set
predict_nb2 = predict(NaiveBayes2,newdata = val)

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confusionMatrixnb2 <- confusionMatrix(predict_nb2,val$verdict)
confusionMatrixnb2

#####################
#SVM Model 
####################

# Training the model
SVM_model1 = svm(verdict~.,data=train)

# Predicting the validation set based on the model
predict_svm1<- predict(SVM_model1,newdata=val)

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confusionMatrixSVM1<- confusionMatrix(predict_svm,val$verdict)
confusionMatrixSVM1

# Reducing the dimenstionality and training again
SVM_model2<- svm(verdict~action+back+bad+character+good+great+worth+perfect+performance+
                 love+scene+make+people+story+time+worst+special+nice+interest+fun+humor+
                 classic+boring+extremely+entertaining+evil+experience+hell, 
               data = train)

# Applying the dimenstionality model into the validation data set
predict_svm2<- predict(SVM_model2,newdata=val)

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confusionMatrixSVM2<- confusionMatrix(predict_svm2,val$verdict)
confusionMatrixSVM2

matrix_backup<-matrix
matrix <- matrix[-which(names(matrix) == "verdict")]

########################
# XGBoost 
########################

# creating the matrix for training the model 
ctrain <- xgb.DMatrix(Matrix(data.matrix(train[,!colnames(train) %in% c('verdict')])), label = as.numeric(train$verdict)-1)

#advanced data set preparation 
dtest <- xgb.DMatrix(Matrix(data.matrix(val[,!colnames(val) %in% c('verdict')]))) 

# setting the watch list with train and validation dMatrixes for the modeling
watchlist <- list(train = ctrain, test = dtest)

# use the xgBoost Modeling training method
xgbmodel <- xgboost(data = ctrain, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20, verbose = 1, watchlist = watchlist)

# predict
predict <- predict(xgbmodel, newdata = data.matrix(val[, !colnames(val) %in% c('verdict')])) 

# see the results of the prediction.
predict.text <- levels(train$verdict)[predict + 1]
table(predict,val$verdict)

# Creating the Confustion Matrix(caret package) for evaluating the metrics
confusionMatrixXG <- confusionMatrix(predict,val$verdict)
confusionMatrixXG


