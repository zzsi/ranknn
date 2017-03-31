# Rankpy: Learning to Rank with Keras/Tensorflow

This python library aims to make it easy to experiment with different models and training algorithms
for a wide range of learning to rank problems, including:

- Sort
- Item/Unit similarity
- Recommendation systems (collaborative filtering, content-based filtering, or hybrid of the two)
- Image quality
- Chatbots (reinforcement learning, deep Q-learning)

Plug in your training data in the form of triplets (see [triplet dataspec](#triplet-dataspec)), and begin to produce models that learn to rank. A working example can be found in the [quickstart](#quickstart) section.

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

Here is an example of applying the library to train a movie recommender model.

First, load MovieLens-100k movie ratings dataset. To see how to generate triplets from this dataset, take a look at `rankpy/data/movielens.py:MovieLensDataset`.

```
from rankpy.data.movielens import MovieLensDataset
from rankpy.models import CollaborativeFilteringModel, HybridRecommenderModel
```

Then train a collaborative filtering recommender.

```
# Trains a collaborative model.
CollaborativeFilteringModel().fit_dataset(MovieLensDataset())
```

In the console, expect to see the the training loss and testing accuracy after 10 training iterations:

```
...
Epoch 9
Epoch 1/1
200/200 [==============================] - 4s - loss: 0.0324
AUC on hold-out test dataset: 0.886500
```

It got an accuracy of 0.89 on test set after 10 training iterations. Not bad as a start..

Now try add more information to the model. To incorporate content-based filtering and include user side information (age, gender, occupation) and item side information (movie genres), you can do:

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

# More advanced usage

Below are some ways to adapt and customize the libary for your own purpose.

## Customize: loss function

You can pick a loss function from two categories: elementwise loss and pairwise loss.

**Pairwise loss**

Minimizing pairwse losses directly optimizes for ranking metrics such as MAP, ROC-AUC, NDCG, mean reciprocal rank.

- Hinge loss
- Logistic

**Elementwise loss (not supported yet)**

Standard regression and classification models belong to this category.

- MSE (linear regression)
- Logistic loss (logistic regression)
- Hinge loss (SVM)
- entropy or gini (for random forest, GBT)

**Connection to matrix factorization and latent factor analysis.** If using elementwise MSE loss, the `rankpy.models.CollaborativeFilteringModel` essentially performs a matrix factorization to solve for a low rank approximation. It does so be constructing one neural net for user, and one neural net for the item, and using the inner product of outputs from the two neural nets as the final output.


## Customize: triplet generator for pairwise loss

You pick a sampling algorithm to generate triplets from the data. The triplet generator works closely with pairwise loss and impacts which subset of training examples to evaluate the pairwise loss on. For example, if you only generate triplets from the top k items of the correctly ranked list, then it focuses on optimizing metrics like NDCG@k.

- totally random sample
- sample from residual (similar to boosting)
	- hard violations: pairs for which the current model makes wrong predictions on their orderings
	- soft violations: pairs for which the current model makes correct predictions but almost misses
	- importance sampling like ada-boost
- sample from top-k items in the correctly ranked list

## Customize: classification or regression model

For elementwise loss, all sklearn, tensorflow, xgboost classification and regression models can be plugged in.

For pairwise loss, the model needs to be differentiable w.r.t. model parameters. This rules out decision tree based models. But there is the neural random decision forests (ICCV 2015 best paper) that can be a good alternative. If you implement your model as a Keras model, it will be readily supported.

## Comparison with current state-of-the-art libraries

It is the plan to include wrappers for popular learning to rank libraries:

- SVM-Rank
- LambdaMart


## Triplet Dataspec

To make use of the training algorithms, the only requirement is to implement a class that has a `triplet_batches` method that generates triplets and a few utility methods to extract metadata from the dataset.

Each *triplet* is consisted of:

- Query (or User)
	- an integer query/user id (already preprocessed by sklearn.preprocessing.LabelEncoder)
	- a vector of content-based features
- Positive item
	- an interger item id
	- a vector of content-based features
- Negative item
	- an iterger item id
	- a vector of content-based features

Query/User id and item id need to be provided if only collaborative filtering is used. The content-based feature vector needs to be provided if content-based filtering is needed. Both types of inputs are needed if a hybrid recommender approach is used.

### Ratings matrix as triplets

A ratings matrix is a matrix with shape (num_users, num_items), with each element of the matrix being a numerical rating score. To generate triplets from a ratings matrix, one can perform a row-wise operation and generate (row_id, positive_item_id, negative_item_id) where row_id corresponds to user_id, and the pair of items satisfy that the item with positive_item_id has a higher rating than the item with negative_item_id.

Optionally, one can append content-based features to the users and items.

### VR Searches as triplets

Here we assume a search contains the user query, a list of search result listings, each tagged with a utility score. Such utility score can come from conversion events. For example, one may define a listing has a utility score of 30 if it is booked as a result of the search, a score of 15 if a booking request is sent, 2 if a click is resulted, etc.

Then a generated triplet can be (query_content_vector, positive_result_content_vector, negative_result_content_vector), where query_content_vector is a numerical vector that encodes the query-dependent features such as query terms, search filters, positive_result_content_vector and negative_result_content_vector are feature vectors for the relatively better and relatively worse search results, according to their utility scores.

### Unit similarity matrix as triplets

Assume we want to train a model according to an existing unit similarity matrix of shape (num_units, num_units).

Triplets can be generated in a very similar way as from a ratings matrix, in a row-wise procedure. Both the query and the item are VR units.

### Image quality ratings as triplets

The dataset has N images, each tagged with a numeric rating. Triplets from this dataset is a special case of more general triplets. A generated triplet is simply (null, positive_image, negative_image).


