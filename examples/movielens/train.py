from rankpy.models import CollaborativeFilteringModel
from rankpy.data import movielens


model = CollaborativeFilteringModel(latent_dim=100, n_epochs=10)
model.fit(movielens)