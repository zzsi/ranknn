from rankpy.models import CollaborativeFilteringModel, HybridRecommenderModel
from rankpy.data import movielens


CollaborativeFilteringModel(latent_dim=100, n_epochs=10).fit_module(movielens)
# HybridRecommenderModel().fit_module(movielens)

