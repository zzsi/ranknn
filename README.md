# Learning to Rank with Tensorflow

A python library to train machine models using learning-to-rank losses in Tensorflow. 
This library curates training algorithms from open source projects and recent academic publications for learning-to-rank related problems. Such training algorithms are very generic and can be used in

- Sort
- Item/Unit Similarity
- Collaborative filtering, content-based filtering or hybrid recommender
- Image quality
- Chatbots



## Install


```
git clone https://github.wvrgroup.internal/zsi/rankpy.git
cd rankpy
pip install -e .
```

or

```
pip install git+ssh://git@github.wvrgroup.internal/zsi/rankpy.git
```
(You need to follow the FAQ section of [data science env wiki](https://wiki.homeawaycorp.com/display/Hostingopps/DataScience+Analysis+Environment), to set up ssh access to internal github.)


## Quickstart

Load MovieLens-100k movie ratings dataset, and train a collaborative filtering recommender.

```
# train.py


from rankpy.data.movielens import MovieLensDataset
from rankpy.models import CollaborativeFilteringModel, HybridRecommenderModel

# Trains a collaborative model.
CollaborativeFilteringModel().fit_dataset(MovieLensDataset())
```

And you would see the the training loss and testing accuracy after 10 training iterations:

```
...
Epoch 9
Epoch 1/1
200/200 [==============================] - 4s - loss: 0.0324
AUC on hold-out test dataset: 0.886500
```


If you like to incorporate content-based filtering and include user side information (age, gender, occupation) and item side information (movie genres), you can do:

```
HybridRecommenderModel().fit_dataset(MovieLensDataset())
```

Result:

```
Epoch 9
Epoch 1/1
200/200 [==============================] - 15s - loss: 1.2094e-06
AUC on hold-out test dataset: 1.000000
Loss on hold-out test dataset: 0.000001
```

## Triplet Dataspec

To make use of the training algorithms, the only requirement is to implement a class that has a `triplet_batches` method that generates triplets and a few utility methods to extract metadata from the dataset.

Each *triplet* is consisted of:

- User (or Query)
	- an integer user id (already preprocessed by sklearn.preprocessing.LabelEncoder)
	- a vector of content-based features
- Positive item
	- an interger item id
	- a vector of content-based features
- Negative item
	- an iterger item id
	- a vector of content-based features

User id and item id need to be provided if only collaborative filtering is used. The content-based feature vector needs to be provided if content-based filtering is needed. Both types of inputs are needed if a hybrid recommender approach is used.

### Ratings matrix as triplets

A ratings matrix is a matrix with shape (num_users, num_items), with each element of the matrix being a numerical rating score. To generate triplets from a ratings matrix, one can perform a row-wise operation and generate (row_id, positive_item_id, negative_item_id) where row_id corresponds to user_id, and the pair of items satisfy that the item with positive_item_id has a higher rating than the item with negative_item_id.

Optionally, one can append content-based features to the users and items.

### VR Searches as triplets

Here we assume a search contains the user query, a list of search result listings, each tagged with a utility score. Such utility score can come from conversion events. For example, one may define a listing has a utility score of 30 if it is booked as a result of the search, a score of 15 if a booking request is sent, 2 if a click is resulted, etc.

Then a generated triplet can be (query_content_vector, positive_result_content_vector, negative_result_content_vector), where query_content_vector is a numerical vector that encodes the query-dependent features such as query terms, search filters, positive_result_content_vector and negative_result_content_vector are feature vectors for the relatively better and relatively worse search results, according to their utility scores.

### Unit similarity matrix as triplets

Assume we want to train a model according to an existing unit similarity matrix of shape (num_units, num_units).

Triplets can be generated in a very similar way as from a ratings matrix, in a row-wise procedure.

### Image quality ratings as triplets

The dataset has N images, each tagged with a numeric rating. Triplets from this dataset is a special case of triplets, where there is no query (or user). A generated triplet is simply (null, positive_image_vector, negative_image_vector).

^
## Losses

### Elementwise MSE loss

**Connection to matrix factorization and latent factor analysis.** If using elementwise MSE loss, the `rankpy.models.CollaborativeFilteringModel` essentially performs a matrix factorization to solve for a low rank approximation. It does so be constructing one neural net for user, and one neural net for the item, and using the inner product of outputs from the two neural nets as the final output.

### Pairwise loss

Two types of activation functions can be used for pairwise loss: logistic function and relu (hinge loss).

Pairwise losses directly optimize ranking metrics like AUC.
