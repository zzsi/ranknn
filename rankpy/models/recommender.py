"""
Adapted from: https://github.com/maciejkula/triplet_recommendations_keras
"""
from __future__ import absolute_import

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Lambda, merge, concatenate
from keras.optimizers import Adam
from keras.constraints import unit_norm
from .. import metrics
from sklearn.metrics import roc_curve, auc


def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss


def hinge_triplet_loss(X, alpha=1):
    positive_item_latent, negative_item_latent, user_latent = X
    loss = K.maximum(0.,
        float(alpha) + K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)
    )
    return loss


def build_cf_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_id')
    negative_item_input = Input((1, ), name='negative_item_id')

    # Shared embedding layer for positive and negative items
    # embeddings_constraint=unit_norm() didn't help
    # TODO: try regularization
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_id')

    positive_item_embedding = Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten()(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))

    loss = Lambda(bpr_triplet_loss, output_shape=(1,), name='brp_loss')(
        [positive_item_embedding, negative_item_embedding, user_embedding]
    )

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(lr=0.001))

    return model


def build_content_model(user_content_dim, item_content_dim, latent_dim):
    pass


def build_hybrid_model(num_users, num_items, user_content_dim, item_content_dim,
    cf_layer_shapes, content_layer_shapes, hybrid_layer_shapes):
    pass


def build_hybrid_model_simple(num_users, num_items, user_content_dim, item_content_dim, latent_dim_cf, latent_dim_hybrid):
    # Inputs.
    user_id_input = Input((1, ), name='user_id')
    user_content_input = Input((user_content_dim, ), name='user_content')
    pos_item_id_input = Input((1, ), name='positive_item_id')
    pos_item_content_input = Input((item_content_dim, ), name='positive_item_content')
    neg_item_id_input = Input((1, ), name='negative_item_id')
    neg_item_content_input = Input((item_content_dim, ), name='negative_item_content')

    user_cf_embedding_model = Sequential()
    user_cf_embedding_model.add(Embedding(num_users, latent_dim_cf, name='user_embedding', input_length=1))
    user_cf_embedding_model.add(Flatten())
    
    item_cf_embedding_model = Sequential()
    item_cf_embedding_model.add(Embedding(num_items, latent_dim_cf, name='item_embedding', input_length=1))
    item_cf_embedding_model.add(Flatten())

    def hybrid_model(cf_embedding, content_input):
        return Dense(latent_dim_hybrid)(
            concatenate([cf_embedding, content_input])
        )

    user_vec = hybrid_model(
        user_cf_embedding_model(user_id_input),
        user_content_input
    )

    positive_item_vec = hybrid_model(
        item_cf_embedding_model(pos_item_id_input),
        pos_item_content_input
    )
    negative_item_vec = hybrid_model(
        item_cf_embedding_model(neg_item_id_input),
        neg_item_content_input
    )

    loss = Lambda(bpr_triplet_loss, output_shape=(1,), name='brp_loss')(
        [positive_item_vec, negative_item_vec, user_vec]
    )

    model = Model(
        inputs=[user_id_input, user_content_input, pos_item_id_input, neg_item_id_input, pos_item_content_input, neg_item_content_input],
        outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(lr=0.001))
    return model


class CollaborativeFilteringModel(object):
    # Cost function: triplet loss for learning to rank.
    # TODO: also implement an element-wise collaborative filtering model (auto-encoder).

    def __init__(self, latent_dim=100, n_epochs=10):
        self.__latent_dim = latent_dim
        self.__n_epochs = n_epochs

    def fit_dataset(self, dataset):
        np.random.seed(42)

        # Read data
        num_users = dataset.num_users()
        num_items = dataset.num_items()
        # test_matrix = data_module.test_matrix()
        test_batch_x, test_batch_y = dataset.triplet_batches(
            mode='test', batch_size=2000,
            include_user_side_info=False,
            include_item_side_info=False
        ).next()

        model = build_cf_model(num_users, num_items, self.__latent_dim)

        # Print the model structure
        print(model.summary())

        # Sanity check, should be around 0.5
        # print('AUC before training %s' % metrics.full_auc(model, test_matrix))

        for epoch in range(self.__n_epochs):

            print('Epoch %s' % epoch)

            model.fit_generator(
                dataset.triplet_batches(
                    mode='train', batch_size=5000,
                    include_user_side_info=False,
                    include_item_side_info=False
                ),
                epochs=1,
                steps_per_epoch=200
            )

            eval_out = model.predict_on_batch(test_batch_x)
            print('AUC on hold-out test dataset: %.6f' % (1. - np.mean(eval_out > 0.5)))
            # print('AUC %s' % metrics.full_auc(model, test_matrix))


class HybridRecommenderModel(object):
    """Collaborative filtering and content-based filtering (aka side information).
    """

    def __init__(self, latent_dim_cf=100, latent_dim_hybrid=100, n_epochs=10):
        self.__n_epochs = n_epochs
        self.__latent_dim_cf = latent_dim_cf
        self.__latent_dim_hybrid = latent_dim_hybrid

    def fit_dataset(self, dataset):
        np.random.seed(42)

        num_users = dataset.num_users()
        num_items = dataset.num_items()
        user_side_info = dataset.get_user_side_info()
        item_side_info = dataset.get_item_side_info()
        user_content_dim = len(user_side_info.values()[0])
        item_content_dim = len(item_side_info.values()[0])
        
        test_batch_x, test_batch_y = dataset.triplet_batches(
            mode='test', batch_size=10000,
            include_user_side_info=True,
            include_item_side_info=True
        ).next()

        model = build_hybrid_model_simple(
            num_users, num_items, user_content_dim, item_content_dim,
            latent_dim_cf=self.__latent_dim_cf,
            latent_dim_hybrid=self.__latent_dim_hybrid)

        for epoch in range(self.__n_epochs):

            print('Epoch %s' % epoch)

            model.fit_generator(
                dataset.triplet_batches(
                    mode='train', batch_size=5000,
                    include_user_side_info=True,
                    include_item_side_info=True,
                ),
                epochs=1,
                steps_per_epoch=200
            )

            eval_out = model.predict_on_batch(test_batch_x)
            print('AUC on hold-out test dataset: %.6f' % (1. - np.mean(eval_out > 0.5)))
            print('Loss on hold-out test dataset: %.6f' % np.mean(eval_out))

