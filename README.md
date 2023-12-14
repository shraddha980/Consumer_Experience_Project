## Consumer Experience Prediction

The most important asset of a Business is its customers. Customers are the best source of feedback. They use the service or product and then they share their feedback hence it is necessary to monitor if the feedback is positive or negative. We can use their comments and insights into improvements for products and services. Closing the customer feedback loop is making sure you have addressed the feedback, and resolving the issue of acknowledging the positive opinion that your customers have shared. This helps us to create a customer-centric culture of action, where the Business gathers customer feedback and turns it into actionable insights.

Steps taken during the feedback loop might be:

1) Sharing an appropriate response with the customer depending if the feedback is positive feedback or negative feedback.
2) If the number of positive responses is more then a system could be devised to automate the positive responses.
3) If we can automate the response to simple issues or queries then it gives employees time to spend on priority issues.
4) The internal process could be improved according to the feedback received in the internal response.

### Business Requirement

_**Basic Flow:**_
- The business collects and analyzes customer feedback using a sentiment analysis tool that assigns a positive, or negative score to each feedback.
- The business uses the sentiment scores to calculate the overall customer satisfaction and loyalty metrics, such as Net Promoter Score (NPS), Customer Satisfaction Score (CSAT), or Customer Effort Score (CES).
- The business identifies the key drivers of customer satisfaction and loyalty, such as product quality, service delivery, pricing, or support.
- The business uses the insights from the sentiment analysis and the customer satisfaction and loyalty metrics to improve the customer experience, such as by resolving issues, enhancing features, offering discounts, or rewarding loyal customers.
  
_**Postconditions:**_ 

The customer receives a better customer experience and is more likely to repurchase, recommend, or advocate for the business.

_**Exceptions:**_ 

Customer feedback is incomplete, inaccurate, or irrelevant. The sentiment analysis tool fails to capture the true emotion or intention of the customer. Customer satisfaction and loyalty metrics are not aligned with the business goals or customer expectations.

### Problem statement

The Business has a problem where the sales of the business have reduced. The Business identified that the count of feedback it has received is quite high and hence it could be used in multiple ways to understand and resolve the issues at hand. The Business took the first 1000 feedbacks and manually labeled them to 1 to "Positive" customer experience and "0" to negative customer experience. 

However, the business has had many such reviews in the past data history and does not have the manpower to label the datasets accordingly. Manually labeling the feedback reviews is a time-consuming process. Hence business needs a solution to automate this process.

The solution is provided in the form of a code that cleans the text, runs a model, and automatically labels them as 0s and 1s. 0 referring to a negative sentiment and 1 referring to a positive sentiment.

### Data Preparation

A labeled dataset of an initial 1000 reviews is taken. Exploratory Data Analysis is performed on this dataset. The following steps are carried out in the EDA process:

1) Checking the total shape of the dataset
2) Checking duplicate and null values
3) Removing the duplicate and null values
4) Removing Stopwords
5) Removing Punctuation Marks
6) Converting the words into lowercase format

Once the data cleaning steps are completed we move towards Label Encoding and Text Vectorization.

### Feature Engineering

We follow the below steps for Feature Engineering:

1) Converting Labelled data into 0 and 1
2) Applying CountVectorizer, Bi-Grams, Tri-Grams, and TF-IDF for converting clean text into integers.

Note: Before applying feature engineering we split the data into Train and Test sets in the ratio of 0.2

### Modelling

We train the data on models Naive Bayes, Decision Trees, Random Forest Classifier, and Support Vector Classifier.

### Evaluation

We check the model using Accuracy_Score using the metrics method from the sklearn library.

## Future Scope

The future scope of this project is that as the amount of data increases we can use deep learning algorithms to train the model.

