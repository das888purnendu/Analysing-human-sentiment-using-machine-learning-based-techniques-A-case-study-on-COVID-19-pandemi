# Analysing-human-sentiment-using-machine-learning-based-techniques-A-case-study-on-COVID-19-pandemi


Abstract-

Sentiment analysis is a very interesting filed for research-based project work. It has significant importance in e-commerce and many other public fields. There are many resources available online to collect public data. Sentiment analysis from tweets has attracted more attention nowadays due to the COVID-19 pandemic situation. Twitter is a part of our regular life. Twitter is used as a micro-blogging site where people can post and interact with each other, it is the best platform to collect the audience’s emotions. Sentiment Classification seeks to identify a piece of text according to its author’s general feeling toward their subject, be it positive or negative. Traditional machine learning techniques have been applied to this problem with reasonable success. In this project work, we present data pre-processing with help of predefined lexicons and labelled with sentiment polarity and build a supervised machine learning model from word count vector with its polarity. We have used several supervised techniques like the Naive Bayes technique, Random forest technique, Decision Tree technique and Logistic regression technique. We have analysed the outcome result of every method from two different datasets. Dealing with tweets many circumstances were present like Emojis, Emoticons, Digits, Website links, many special characters in a tweet text. It is very difficult to build a high accuracy machine learning model with those things, in this project we have removed the emojis, emoticons, website links, digits, and special characters from the dataset. We got the highest accuracy on the sentiment balanced dataset with the Logistic regression approach.



Introduction -

Sentiment analysis or opinion mining is one of the major tasks of NLP (Natural Language Processing). Sentiment analysis deals with identifying and classifying opinions or sentiments expressed in the source text. Social media is generating a vast amount of sentiment rich data in the form of tweets, status updates, blog posts, etc. Sentiment analysis of this user generated data is very useful in knowing the opinion of the crowd. Human opinions have very important roles in everywhere. In ecommerce, shopper must want to know the buyer’s opinion and the sentiment. And also, many other aspects.
Twitter sentiment analysis is difficult compared to general sentiment analysis due to the presence of slang words and misspellings. The maximum limit of characters that are allowed in Twitter is 140. And also, it has many website links, emojis, emoticons, special characters. Twitter API also have limitations. Knowledgebase approach and Machine learning approach are the two strategies used for analysing sentiments from the text. In this project, we try to analyse twitter posts using the lexicons sentiment polarity & Machine Learning approaches. By doing sentiment analysis in a specific domain, it is possible to identify the effect of domain information in sentiment classification. In this project our domain is COVID-19 pandemic situation and targeted tweets related to #lockdown #india. We present a new feature lexicons sentiment polarity and word count vector for classifying the tweets as positive or negative depends on the polarity.
We examine the effectiveness of applying machine learning techniques to the sentiment classification problem. Our analysis helps concerned organizations to find opinions of people about current pandemic situation from their tweets, if it is positive or negative. The challenging aspect in sentiment analysis is an opinion word which is considered as a positive in one situation may be considered as negative in another situation. The traditional text processing with porter stemming technique may change the meaning of the original sentence. Whereas we can use word lemmatization for better text processing. There are different algorithms that can be used in the stemming process, but the most common in English is Porter stemmer. The rules contained in this algorithm are divided in five different phases numbered from 1 to 5. The purpose of these rules is to reduce the words to the root. Lemmatization is the key to this methodology is linguistics. To extract the proper lemma, it is necessary to look at the morphological analysis of each word. This requires having dictionaries for every language to provide that kind of analysis.
In this project report we have demonstrate how the dataset is pre-processed and lemmatizing. How to determine the sentence polarity with help of lexicons and update the dataset with the sentence polarity. How much a machine learning model’s accuracy depends on the sentiment polarity of the maximum number of the data. Improvement of different machine learning models. And select the best model among them. Finally result analysis and practical implementation.


Objective -

This project is implemented of sentiment classification of a tweet and build a best model that can helps to predict the sentiment with high accuracy. In this project our aims are:
• Build supervised unbiased model.
• Choose the best model.
• Increase the model’s precision value.
• Increase the model’s accuracy.
• Make a record of the model with real example text.


Methodology -

In this project work we have made 5 different phases. Which are
1. Data collecting through twitter API.
2. Data cleaning.
3. Data pre-processing with help of POS tagging
4. Determine the sentiment polarity via Lexicons and make labelling.
5. Build supervised model
Tools are used:
1. Python libraries:
I. NLTK [NLP tools]
II. TextBlob [Text processing tools]
III. SKlearn [Machine Learning tools]
IV. Pandas [Dataset handling tool]
V. Matplotlib [Data visualization tools]
VI. Tweepy [Twitter API tools]
VII. Re [Regular expression tools]
VIII. Numpy [array & math tools]
2. Twitter API




Result -

Logistic Regression model :
Accuracy: 0.9620283018867924
Precision: 0.9570049722140976
Recall: 0.9492312155497534
Log loss: 1.3115098514131467
F1 Score: 0.9531022429362075
AUC: 0.9600119916413579
