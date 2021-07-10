# script to download all datasets, since it currently doesn't work to download them all at once

import minerl
import argparse
import datasets

ENVS = datasets.ENVS

def main(data_dir):

    for i, env in enumerate(ENVS):
        print(f'Now downloading {i+1}/{len(ENVS)}')
        minerl.data.download(data_dir, environment=env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir')

    args = vars(parser.parse_args())

    main(**args)

    