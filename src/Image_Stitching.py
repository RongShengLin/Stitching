from Stitching import Stitch
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-ord", "--order", help="Order of Images", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("input", help="Input Image list")
parser.add_argument("-m", "--mask", help="Mask size of linear blending with constant width.", type=int, default=-1)
parser.add_argument("-o", "--output", help="Output directory", default="./result")
parser.add_argument("-f", "--feature", help="Show feature images.", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("-fm", "--feature_match", help="Show feature match images.", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()
input_list, order, mask, output_dirname, show_feature, show_match = args.input, args.order, args.mask, args.output, args.feature, args.feature_match

if ord not in {-1, 1}:
    ord = 1

tag = 1
if mask <= 0 or mask > 70:
    print("Use default mask.")
    mask = -1

if not os.path.isfile(input_list):
    print("File not exist: ", input_list, file=sys.stderr)
    exit(0)

if os.path.exists(output_dirname):
    if not os.path.isdir(output_dirname):
        print(output_dirname, "exists and not a directory.")
        output_dirname = "./"
else:
    print(output_dirname, "not exists, create one.")
    os.makedirs(output_dirname)
print("Will save output images to", output_dirname)

feature_dir = os.path.join(output_dirname, "features")
if show_feature or show_match:
    if os.path.exists(feature_dir):
        if not os.path.isdir(feature_dir):
            print(feature_dir, "exists and not a directory.")
            feature_dir = output_dirname
    else:
        print(feature_dir, "not exists, create one.")
        os.makedirs(feature_dir)
    print("Will save feature images to", feature_dir)


Stitch(input_list, order, mask, output_dirname, show_feature, feature_dir, show_match)