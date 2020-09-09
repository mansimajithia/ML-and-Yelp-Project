# De-Mistyfing Machine Learning through Restaurant Data

# Data Cleaning and Exploration
Data Source: https://www.yelp.com/dataset
Jupyter Notebook: https://colab.research.google.com/drive/1I9bEY_uvQ-LCv7jTvYK9EIyTmzyaS8Lx

Our datasource  came from Yelp Open Data. This dataset is a subset of data from 25 states and 8 provinces in Canada from 2018. The original JSON file contains 209,000 rows of records. Since we just wanted to focus on Restaurant Data, we cleaned out the data and filtered to restaurants that contained ‘Restaurant’ under categories. We ended up with 180,000 rows of restaurant data. Next we only wanted restaurants that included the attributes we were looking for. Our final dataset was with 21,000 rows.

There were a lot of attributes, so we discarded like the ambiance of a restaurant. We didn’t think whether a ‘hipster’ ambiance would have effect on the restaurant’s future. Many of the attributes had Boolean value which we assigned as 1 or 0. 

There were certain attributes that didn’t have Boolean values such as type of parking, number of reviews, restaurant rating, or restaurant price. For the parking, we turned those into different columns and then one-hot encoded accordingly

For number of reviews we first got the median and then binned them by their quartiles and then dummified each quartile. 

Based on our research on restaurant success, we found that restaurant density can also play an important role. We logged the number of restaurants in a half mile radius of each restaurant and dummified it the same  way we dummied number of reviews, through quartiles and binning.

Now that we had each feature one-hot encoded, we were ready for Machine Learning. But we also wanted to visually see if there were any correlation between each feature or if there were any trends geographically or otherwise. We used Tableau for this type of analysis.

# Tableau

We determined that a visual outlook using Tableau would better determine our approach for Machine Learning. One of our goals was to observe frequency and distribution of features across locations. The other was to confirm certain correlations in our feature. And finally to compare feature data between open and close restaurants. And also to run a sentiment analysis of our dataset.

Our data was broken down by ratings, price range, along with our dummied features. By looking through the Tableau Data, you will see visualizations based on geographical locations, visualizations broken down by prices and ratings. 

Some observation:

-Open restaurants that sold alcohol had higher rating

-Closed restaurants that had outdoor seatings had lower rating

-Closed restaurants also had fewer ratings

-Restaurants that had the highest and lowest had the highest rating count. They also had more useful rating count.

# Machine Learning

Jupyter Notebook: https://colab.research.google.com/drive/12KDRU3iXYiUAHk29lg839A4cY_hmBIMM?usp=sharing

We had 14 features to train and test the model

We performed a logistical regression which consistently gave us a result of 0.79

We also a performed a Random Forest Classifier which had a result of 0.756. This number included our restaurant density. Prior to restaurant density, we had a score of 0.7172. 

Despite running our logistical regressing on multiple iterables, our score did not change much.

We also ran other Machine Learning Models on our dataset. Our scores were consistently around 0.7  to 0.79

We observed our feature importances in random classifier and logistic regression. 
Wifi and Outdoor Seating were more important while validated parking were least important.

We also created a correlation heatmap. Dark red color indicates a significant negative correlation between features. Dark Green color indicates a significant positive correlation.

We also wanted to see which model, logistical regression or Random Forest Classifier is the “better model” to predict restaurant success. We built a confusion matrix model. In creating a confusion matrix model, we found that each model was good at predicting restaurants that would remain open with each being above 90%, but neither did a good job at predicting which model would do a good job fail. Each showed a high propensity of false positive or that over predicting number of restaurants that would remain open. Random Forest Classifier had a 76% positive while Log Reg at 89% False Positive Percent. Therefore we determined that Random Forest had a better predictive model.

We determined that the nature of high false negative was due to the fact that Yelp drops closed restaurants after 3 months thus eliminating them from our training model.
