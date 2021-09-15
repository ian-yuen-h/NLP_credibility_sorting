import argparse


def run_pipeline(dataset, n):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Media Credibility Prediction'
    )
    parser.add_argument('--dataset', type=str, required=True, help=(
        'Name of dataset to extract articles from.')
    )
    parser.add_argument('--n', type=str, required=True, help=(
        'Number of articles to search')
    )
    args = parser.parse_args()

    run_pipeline(args.dataset, args.n)