# adapted from https://gist.github.com/hysts/81a0d30ac4f33dfa0c8859383aec42c2

import argparse
import cv2
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    event_acc = event_accumulator.EventAccumulator(
        args.event_path, size_guidance={'images':0})
    event_acc.Reload()

    os.makedirs(args.save_dir, exist_ok=True)

    for tag in event_acc.Tags()['images']:

        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = os.path.join(args.save_dir, tag_name)
        os.makedirs(dirpath, exist_ok=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = os.path.join(dirpath, '{:04}.jpg'.format(index))
            cv2.imwrite(outpath, image)


if __name__ == '__main__':
    main()