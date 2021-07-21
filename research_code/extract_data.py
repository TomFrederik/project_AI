from datasets import extract_data, ENVS
import argparse

def main(args):
    # skip envs that are already downloaded 
    # to not having to run everything again while still debugging
    skipenvs = ['MineRLNavigateExtremeVectorObf-v0',
        'MineRLNavigateExtremeDenseVectorObf-v0']
    for env in ENVS:
        if env in skipenvs:
            continue
        args['env_name'] = env
        extract_data(**args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_samples', default=0)
    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir')
    #parser.add_argument('--num_workers', default=2, type=int)

    args = vars(parser.parse_args())

    main(args)

    