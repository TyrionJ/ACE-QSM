import argparse
from scripts.preprocessor import Processor

pf = '../../data/ACE-QSM_processed'
rf = '../../data/ACE-QSM_raw'


def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=pf, help='Processed folder')
    parser.add_argument('-r', type=str, default=rf, help='Raw folder')
    parser.add_argument('-D', type=int, default=1, help='Dataset ID')
    args = parser.parse_args()

    Processor(dataset_id=args.D, raw_folder=args.r, processed_folder=args.p).run()


if __name__ == '__main__':
    run_preprocess()
