from rankpy.models import SimpleRecommender
from rankpy.data import movielens


model = SimpleRecommender()
model.fit(movielens)