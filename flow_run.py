import argparse
import itertools
import os
import yaml
from optical_flow import lucas_kanade, farneback

colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
    'yellow': [0, 255, 255], 'white': [255, 255, 255]}

parser = argparse.ArgumentParser(description='Calculate optical flow.')
parser.add_argument('image1', type=str, help='Input image No1.')
parser.add_argument('image2', type=str, help='Input image No2.')
parser.add_argument('--output_path', '-o', type=str, default='flow/', help='path to output directory')
parser.add_argument('--method', '-m', type=str, default='lk', choices=['lk', 'fb'], help='Select a method.')
parser.add_argument('--circle_color', '-cc', type=str, default='yellow', choices=colormap.keys(), help='Select a color for circle.')
parser.add_argument('--line_color', '-lc', type=str, default='red', choices=colormap.keys(), help='Select a color for line.')
parser.add_argument('--vector_scale', '-vs', type=float, default=60.0, help='Scale saving vector data.')
parser.add_argument('--circle_size', '-s', type=int, default=2, help='Size of original point marker.')
parser.add_argument('--line', '-l', type=int, default=2, help='Width of vector line.')
args = parser.parse_args()

if __name__ == "__main__":

    if args.method == 'lk':
        lucas_kanade(args.image1, args.image2, args.output_path, args.vector_scale,
            args.circle_size, args.line_color, args.line, args.circle_color)
    elif args.method == 'fb':
        farneback(args.image1, args.image2)