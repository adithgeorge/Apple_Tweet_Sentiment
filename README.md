# Apple_Tweet_Sentiment

**Still working on this by improving the result, gaining more data and further improvement in deployment. Do read this file and check the conclusion to get an accurate idea on my project and what are its limitations.**

**You can access the web app through this link:**

https://apple-tweet-sentiments.herokuapp.com/

**There is a problem found when connecting to streamlit on a chromium browser on PC. UPDATE: Use incognito mode.** 
**It is better to open the web app via a mobile browser or using Microsoft Edge on PC.**

## Problem Statement

In the following notebook we are going to be performing sentiment analysis on a collection of tweets about Apple Inc. After preprocessing, the tweets are labeled as either positive (i.e. I love the new iMac) or negative. (i.e. Apple has bad work poilicies!)

## Data Description

* Tweet Sentiment: Positive Sentiment = 1, Negative Sentiment = 0
* Sentiment Confidence: Range of (0,1) describing the confidence of the sentiment assignment.
* Text: Text composition of the tweet

## Feature Engineering

* The raw data comes with a lot of information that is not necessary for the analysis. In this section we toss out these sections.

* Here we can see that some of the tweets are neutral in nature represented by '3'. We are not considering neutral tweets for our model for now but can be used in further scaling of this project.

* We see that there are times as much negative tweets in the data than positive tweets. The model accuracy and performance will increase if we could collect more data for training.

## Preprocessing 

* Because the data is made up of Tweets, there are many non-text symbols (i.e @, https:, # etc.) throughout our dataset.We will keep some symbols that provide semantics (i.e !,',"). This will only serve to confuse our model and make sentiment predictions less accurate. So we choose to make a simpler and potentially more accurate model by removing those symbols.

### Data Tokenisation 

* Here we turn our twitter strings to lists of individual tokens. (words, punctuations)

* We have an average sentence length around 14-15 words, and a vocabulary size of 3,701 unique words.

### TF-IDF

* We use TF-IDF to convert our token lists to numerical data.

## Model Building 

So now we predict the whether the tweet sentiment is positive or not using Classification Algorithms. Following are the algorithms I will use to make the model:

1. Logistic Regression

2. Support Vector Machines (Linear)

3. Random Forest

4. K-Nearest Neighbours

5. Decision Tree

The accuracy of a model is not the only factor that determines the robustness of the classifier. As the training and testing data changes, the accuracy will also change. It may increase or decrease. This is known as model variance.

To overcome this and get a generalized model, we use Cross Validation.

Many a times, the data is imbalanced, i.e there may be a high number of one specific class instances but less number of other class instances. Thus we should train and test our algorithm on each and every instance of the dataset. Then we can take an average of all the noted accuracies over the dataset.

An algorithm may underfit over a dataset for some training data and sometimes also overfit the data for other training set. Thus with cross-validation, we can achieve a generalised model.

* **As we can see the best performing algorithms are SVM (Linear), Random Forest Classifer, Logistic Regression and KNN.**

* Cross Validation is a type of resampling method and it should be done on the complete data to know how our model will work when subjected to different data.

### Cross Validation

* Here, we can see that the Linear SVM, KNN, Logistic Regression and Random Forest are performing well enough after cross validation.
* **Linear SVM is having a higher mean accuracy score.**
* From the cross validation scores we can see that there is enough variance in the performance of all models with different data and therefore we can assume that performance of model is subjective to data.

### Model Evaluation 

* We can get a summarized result with the help of a classification report and confusion matrix, which shows where did the model go wrong, or which class did the model predict wrong. They give the number of correct and incorrect classifications made by the classifier.

From the confusion matrix we can infer the following:

* The model as it was trained with more negative samples has a bias towards negative tweets. There is higher chance of prediction to be negative.

* It can be seen there are less number of samples predicted as positive.

* In all the models, there is high False Negative prediction which means that the model is not very adept at predicting positive tweets.

* We need a model with less False Negative and False Positive predictions and therefore we can choose SVM model to be better than others
There is a a need to train the model with more data to improve the model accuracy in the future.

### Hyper Parameter Tuning

Hyper Parameter Tuning can be done to change the learning rate of the algorithm and get a better model. The follwoing models can be tuned:

1. Linear SVM
2. KNN
3. Random Forest Classifier

* **From the hyper parameter tuning, we can see that all the models perform well but Linear SVM is having a slight advantage when it comes to overall accuracy.**

### Boosting

It is an ensemble technique used for the step by step enhancement of a weak model.

* A model is first trained on the complete dataset. Now the model will get some instances right while some wrong. Now in the next iteration, the learner will focus more on the wrongly predicted instances or give more weight to it. Thus it will try to predict the wrong instance correctly. Now this iterative process continous, and new classifers are added to the model until the limit is reached on the accuracy.

* We can observe that the SVM model when hypertuned still work better than the boosting models. So we will stick to it.

# Conclusion

* We built a linear model that predicts the sentiment of tweets about Apple at around 83% accuracy.
* The Confusion Matrix showed a tendency towards false negatives. Lastly we showed that the model succesfully inferred the importance of some english words to twitter sentiment.
* More and better data is needed to improve the model accuracy and also reduce it's bias to negative tweets.

