import argparse
import platform
from scripts.preprocessor import Processor

if platform.system().lower() == 'windows':
    pf = r'F:\Data\runtime\ACE-QSM\ACE-QSM_processed'
    rf = r'F:\Data\runtime\ACE-QSM\ACE-QSM_raw'
else:
    pf = '/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_processed'
    rf = '/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw'


def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=pf, help='Processed folder')
    parser.add_argument('-r', type=str, default=rf, help='Raw folder')
    parser.add_argument('-D', type=int, default=8, help='Dataset ID')
    args = parser.parse_args()

    Processor(dataset_id=args.D, raw_folder=args.r, processed_folder=args.p).run()


if __name__ == '__main__':
    run_preprocess()
