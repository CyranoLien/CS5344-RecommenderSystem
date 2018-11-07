# CS5344-RecommenderSystem

Data source:
http://jmcauley.ucsd.edu/data/amazon/links.html  
 
Please download '5-core' and 'meta' for 'Apps_for_Android'.  

The former is the subset of the data in which all users and items have at least 5 reviews.  
 
The latter contains lists of 'also_bought' and 'also_viewed' for each specific product. 

To evaluate the methods, we compare the Conversion Rate (CR) of the recommendations. 
A user has obtained at least one good recommendation if s/he purchased at least one product from the recommended list of top K items.
 

## Frequent Item Sets

## Collaborative Filtering
The method used here is item-based CF. 
 
Note that, the testing set here must be sampled from those transactions where score=5.  

That is, in the original dataset, a customer may score a product very low but still he/she bought it, so we have to re-think how to define 'conversion' ourselves.
A reasonable method is, if a customer is predicted to highly score some products we recommend those products.  

Thus, we have to sample those transactions whose score is as high as possible (5 actually), so that we could regard the customer did buy the product.
Now the task is changed to find whether 'a customer would highly score one indeed from the products we provide'.

1. Prepare the data of 5-core so that all users and items have at least 5 reviews.

2. Convert the data from string to numerical for the sake of using Spark MLlib.

3. Split the dataset into training and testing. Note that, for testing data, we only pick those highly-scored transactions. 

4. Transform the testset. Make a list with the size being M*N. (M refers to the numbers of customers in the testset, and N for products).

5. Train the model using explicit method.

6. Predict for testset, and for each customer pick K items. Compare whether the highly-scored-indeed item is in the list.

7. May aslo simply split a testset, predict the socores and compare RMSE.


