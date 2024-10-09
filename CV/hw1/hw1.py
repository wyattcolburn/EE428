import rawpy
import argparse

# Parse command line arguments.
# Usage:
#    python hw1.py <DNG filename>
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

# read 16-bit integer data from RAW file into raw_image array
with rawpy.imread(args.path) as raw:
    raw_image = raw.raw_image.copy()

# filename for JPEG output
path_out  = args.path.split('.')[0] + '.jpg'