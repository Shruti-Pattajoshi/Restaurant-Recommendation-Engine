# Restaurant-Recommendation-Engine
Recommender Engines or Systems are among the most popular applications of data science today. They are used to predict the “rating” or “preference” that a user would give to an item. 
Recommender Systems can be of three different types:

1. Simple Recommenders : Here, the top or head of the data frame (after being sorted in the decreasing order of their ratings) will give the most popular restaurants which will most likely be liked by everyone. Since the number of votes for all the restaurants wont be same, so you cant directly compare that way: thus Weighted Rating is calculated and the top restaurants is projected.

2. Content-based Recommenders: Here, the basic idea is that if a person liked a particular restaurant, then he or she will also like a restaurant that is similar to it thus suggests similar restaurants
This basically computes the similarity between restaurants based on certain parameter and suggests restaurants that are most similar to a particular restaurant that a user liked previously.
The restaurants are given similarity scores on the basis of its features ( using TfidfVectorizer on the features and then finding the similarity scores by the Cosine Similarities - faster after using TfidfVectorizer ) . Later when given a restaurant as input the top/head() of the data frame agregated on the similarity scores to that restaurant will be displayed.

3.Collaborative filtering engines:  This type of recommenders try to predict the rating that a user would give an restaurant-based on past ratings and preferences of other users. The users rate the restaurants and the similarity b/w users is found to suggest one with the restaurants highly liked by the other similar user. There are again 2 types : User Based and Item Based.

I used Yelp Restaurant Dataset for the engine : User_Review(u-r) and Business_Review(r-count) datasets followed by filtering of less popular restaurants and top 10 cities to reduce the data.The two dataset are transformed into sparse matrix and then using the KNN with Cosine Similarity b/w restaurants, it displays the most similar restaurants to any user input-restaurant. This is Item Based Filtering.

Challenges:
Popularity bias: refers to system recommends the movies with the most interactions without any personalization
Item cold-start problem: refers to when movies added to the catalogue have either none or very little interactions while recommender rely on the movie’s interactions to make recommendations
Scalability issue: refers to lack of the ability to scale to much larger sets of data when more and more users and movies added into our database

User Based Filtering :

In this case of collaborative filtering, matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. One matrix can be seen as the user matrix where rows represent users and columns are latent factors. The other matrix is the item matrix where rows are latent factors and columns represent items.




For each user, first pool all the reviews together to form a single paragraph and then apply TFIDF Vectorizer from scikit learn package to extract the features from the text for both the business_reviews and user_reviews datasets
After all, we got the feature vectors P for user Id and Q for business Id.
Now after that created a user item rating matrix by using the attributes user_id, business_id, rating_stars. Then the user-item into factorized into two : by Matrix Factorization is to decompose each user rating into a User-Factor Vector and A Product-Vector.
Call the matrix_factorization method with all the hyper parameter.
The most relevant restaurant based on the user search is predicted i.e. simply the inner product of the feature vector of plain text and feature vectors of business Id. The top N records fetched and displayed.





