<32>

## Title
Content-based Filtering

## Description
Recommends items based on the features of users and features of an item having a good match

Still have data where users have rated an item i

Train neural network for each user and each item using user features and item features respectively.
Take the dot product of the user network output layer and the item network output layer to produce a prediction for the user's rating of the item. (can also be implemented with the sigmoid function for boolean prediction)

## Additional Notes

![[content_based_filtering.png]]
![[Pasted image 20221114183204.png]]
![[Pasted image 20221114183228.png]]
![[Pasted image 20221114183302.png]]
![[Pasted image 20221114183407.png]]
User features (examples):
- age
- gender
- country, 
- average rating (of the user) per genre, 
- movies watched

Movie features (examples):
- year of movie
- genre/genres
- reviews
- Average rating (from users)

## Linked Cards


## Tags
[[Machine Learning]] 