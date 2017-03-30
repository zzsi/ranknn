import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='cf', choices=['cf', 'hybrid'],
        help=('Collaborative filtering (cf) or Hybrid of Collabroative' +
              'filtering and Content-based filtering (hybrid)')
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    from rankpy.models import CollaborativeFilteringModel, HybridRecommenderModel
    from rankpy.data.movielens import MovieLensDataset
    if args.method == 'cf':
        CollaborativeFilteringModel(latent_dim=100, n_epochs=10).fit_dataset(MovieLensDataset())
    elif args.method == 'hybrid':
        HybridRecommenderModel().fit_dataset(MovieLensDataset())
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
