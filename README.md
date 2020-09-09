# De-Mistyfing Machine Learning through Restaurant Data

# Data Cleaning and Exploration
Data Source: https://www.yelp.com/dataset
Jupyter Notebook: https://colab.research.google.com/drive/1I9bEY_uvQ-LCv7jTvYK9EIyTmzyaS8Lx

Our datasource  came from Yelp Open Data. This dataset is a subset of data from 25 states and 8 provinces in Canada from 2018. The original JSON file contains 209,000 rows of records. Since we just wanted to focus on Restaurant Data, we cleaned out the data and filtered to restaurants that contained ‘Restaurant’ under categories. We ended up with 180,000 rows of restaurant data. Next we only wanted restaurants that included the attributes we were looking for. Our final dataset was with 21,000 rows.

There were a lot of attributes, so we discarded like the ambiance of a restaurant. We didn’t think whether a ‘hipster’ ambiance would have effect on the restaurant’s future. Many of the attributes had Boolean value which we assigned as 1 or 0. 
```
#=======================================Convert Attributes to 1/0 Features ==================================================#

restaurantData3= restaurantData2 # make a copy of ResData 3

# Get Reservations
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsReservations':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['reservations'] = df_attribute['RestaurantsReservations']
restaurantData3['reservations'] = restaurantData3['reservations'].astype(str).map({'True': 1, 'False': 0})


# Get Delivery
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsDelivery':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['delivery'] = df_attribute['RestaurantsDelivery']
restaurantData3['delivery'] = restaurantData3['delivery'].astype(str).map({'True': 1, 'False': 0})


# Get Takeout
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsTakeOut':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['takeout'] = df_attribute['RestaurantsTakeOut']
restaurantData3['takeout'] = restaurantData3['takeout'].astype(str).map({'True': 1, 'False': 0})



# Get Price
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsPriceRange2':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['pricerange'] = df_attribute['RestaurantsPriceRange2'].astype(int)



# Get Accept Credit Cards
df_attribute = pd.DataFrame([i if i != None else {'BusinessAcceptsCreditCards':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['acceptcreditcard'] = df_attribute['BusinessAcceptsCreditCards']
restaurantData3['acceptcreditcard'] = restaurantData3['acceptcreditcard'].astype(str).map({'True': 1, 'False': 0})


# Get Outdoor Seating
df_attribute = pd.DataFrame([i if i != None else {'OutdoorSeating':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['outdoorseating'] = df_attribute['OutdoorSeating']
restaurantData3['outdoorseating'] = restaurantData3['outdoorseating'].astype(str).map({'True': 1, 'False': 0})


# Get Good For Groups
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsGoodForGroups':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['goodForGroups'] = df_attribute['RestaurantsGoodForGroups']
restaurantData3['goodForGroups'] = restaurantData3['goodForGroups'].astype(str).map({'True': 1, 'False': 0})


# Get Good For Kids
df_attribute = pd.DataFrame([i if i != None else {'GoodForKids':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['kidsfriendly'] = df_attribute['GoodForKids']
restaurantData3['kidsfriendly'] = restaurantData3['kidsfriendly'].astype(str).map({'True': 1, 'False': 0})
#restaurantData3['kidsfriendly'] = restaurantData3['kidsfriendly'].astype(float).round(2)



# Get happyhour
df_attribute = pd.DataFrame([i if i != None else {'HappyHour':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['happyhour'] = df_attribute['HappyHour']
restaurantData3['happyhour'] = restaurantData3['happyhour'].astype(str).map({'True': 1, 'False': 0})

# BYOB
df_attribute = pd.DataFrame([i if i != None else {'BYOBCorkage':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['byobfee'] = df_attribute['BYOBCorkage']
restaurantData3['byobfee']=restaurantData3['byobfee'].str.contains("no")
restaurantData3['byobfee'] = restaurantData3['byobfee'].astype(str).map({'True': 1, 'False': 0})


# Get Table Service
df_attribute = pd.DataFrame([i if i != None else {'RestaurantsTableService':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['tableservice'] = df_attribute['RestaurantsTableService']
restaurantData3['tableservice'] = restaurantData3['tableservice'].astype(str).map({'True': 1, 'False': 0})


# Get Alcohol
df_attribute = pd.DataFrame([i if i != None else {'Alcohol':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['alcohol'] = df_attribute['Alcohol']
restaurantData3['alcohol'] = restaurantData3['alcohol'].str.contains("none")
restaurantData3['alcohol'] = restaurantData3['alcohol'].astype(str).map({'True': 1, 'False': 0})


# WiFi
df_attribute = pd.DataFrame([i if i != None else {'WiFi':np.NAN} for i in restaurantData3['attributes']])
restaurantData3['wifi'] = df_attribute['WiFi']
restaurantData3['wifi'] = restaurantData3['wifi'].str.contains("free")
restaurantData3['wifi'] = restaurantData3['wifi'].astype(str).map({'True': 1, 'False': 0})

pd.set_option('display.max_columns', None)
```

There were certain attributes that didn’t have Boolean values such as type of parking, number of reviews, restaurant rating, or restaurant price. For the parking, we turned those into different columns and then one-hot encoded accordingly

```
parking_columns=['parking_garage', 'parking_lot', 'parking_street', 'parking_valet', 'parking_validated']
restaurantData3[parking_columns] = cleaned_parking_df

restaurantData3['parking_garage'] = restaurantData3['parking_garage'].astype(float).round(2) # Convert string True/False to float
restaurantData3['parking_lot'] = restaurantData3['parking_lot'].astype(float).round(2) # Convert string True/False to float
restaurantData3['parking_street'] = restaurantData3['parking_street'].astype(float).round(2) # Convert string True/False to float
restaurantData3['parking_valet'] = restaurantData3['parking_valet'].astype(float).round(2) # Convert string True/False to float
restaurantData3['parking_validated'] = restaurantData3['parking_validated'].astype(float).round(2) # Convert string True/False to float
```

For number of reviews we first got the median and then binned them by their quartiles and then dummified each quartile. 
```
category = ['review_37', 'review_82', 'review_184', 'review_max'] # Pass bin name
# Create review_count bins
restaurantData3['category']=pd.cut(x= restaurantData3['review_count'], bins=[0, 37, 82, 184, 10129],labels=category)

# *********Dummified binned review_count**************
review_df = pd.get_dummies(restaurantData3["category"])
review_df.head()

restaurantData3[category] = review_df
# pd.set_option('display.max_columns', None) # To display all columns


# *********Dummified price range**************
price_df = pd.get_dummies(restaurantData3["pricerange"])
price_columns=['price_1', 'price_2', 'price_3','price_4']
restaurantData3[price_columns] = price_df
price_df.head(100)

restaurantData3.head(25)
```

Based on our research on restaurant success, we found that restaurant density can also play an important role. We logged the number of restaurants in a half mile radius of each restaurant and dummified it the same  way we dummied number of reviews, through quartiles and binning.

```
df2=restaurantData3

for index1, row1 in df2.iterrows():
    num_restaurants = 0
    min_lat = row1['latitude']-.07/10
    max_lat = row1['latitude']+.07/10
    min_long = row1['longitude']-.09/10
    max_long = row1['longitude']+.09/10
    df_temp = df2[(df2['latitude']>min_lat) & 
                  (df2['latitude']<max_lat) & 
                  (df2['longitude']>min_long) & 
                  (df2['longitude']<max_long)]
    num_restaurants = len(df_temp)
    df2.loc[index1, 'num_restaurants_1mile'] = num_restaurants - 1 # (subtract 1 to exclude self)"
    
    
df2.head()
```
Now that we had each feature one-hot encoded, we were ready for Machine Learning. But we also wanted to visually see if there were any correlation between each feature or if there were any trends geographically or otherwise. We used Tableau for this type of analysis.

# Tableau
https://public.tableau.com/profile/bosco.sitati#!/vizhome/Project3_15975163220020/Yelp-DataAnalysis-SentimentalAnalysis?publish=yes

We determined that a visual outlook using Tableau would better determine our approach for Machine Learning. One of our goals was to observe frequency and distribution of features across locations. The other was to confirm certain correlations in our feature. And finally to compare feature data between open and close restaurants. And also to run a sentiment analysis of our dataset.

Our data was broken down by ratings, price range, along with our dummied features. By looking through the Tableau Data, you will see visualizations based on geographical locations, visualizations broken down by prices and ratings. 

Some observation:

-Open restaurants that sold alcohol had higher rating

-Closed restaurants that had outdoor seatings had lower rating

-Closed restaurants also had fewer ratings

-Restaurants that had the highest and lowest had the highest rating count. They also had more useful rating count.

![analysis](https://github.com/mansimajithia/ML-and-Yelp-Project/blob/master/images/Line%20Graph.png)

![sentiment](https://github.com/mansimajithia/ML-and-Yelp-Project/blob/master/images/Sentiment%20Anaysis.png)

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

![matrix](https://github.com/mansimajithia/ML-and-Yelp-Project/blob/master/images/matrix.png)

We also wanted to see which model, logistical regression or Random Forest Classifier is the “better model” to predict restaurant success. We built a confusion matrix model. In creating a confusion matrix model, we found that each model was good at predicting restaurants that would remain open with each being above 90%, but neither did a good job at predicting which model would do a good job fail. Each showed a high propensity of false positive or that over predicting number of restaurants that would remain open. Random Forest Classifier had a 76% positive while Log Reg at 89% False Positive Percent. Therefore we determined that Random Forest had a better predictive model.

![confusion](https://github.com/mansimajithia/ML-and-Yelp-Project/blob/master/images/confusion3.png)

We determined that the nature of high false negative was due to the fact that Yelp drops closed restaurants after 3 months thus eliminating them from our training model.
