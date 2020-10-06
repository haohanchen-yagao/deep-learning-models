import os, sys
import boto3
import re
import argparse
from datetime import datetime


def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group(1)
    return found

def extract_result(log_abspath):
    result = []
    for i in range(64):
        result.append(0)
    images = []
    for i in range(5944):
        images.append(0)
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'See image name: ' in line:
                image_idx = int(regex_extract(line, 'See image name: ([-+]?\d*\.\d+|\d+)'))
                gpu_idx = int(regex_extract(line, '([-+]?\d*\.\d+|\d+) 00000'))
                result[gpu_idx] += 1
                images[image_idx] += 1
    for i in range(64):
        print("gpu {} has {} images".format(i, result[i]))
    for i in range(5000):     
        print("img {} has {} images".format(i, images[i]))

def extract(log_abspath):
    result = []
    for i in range(4944):
        result.append(0)
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'placed idx is: ' in line:
                image_idx = int(regex_extract(line, 'placed idx is: ([-+]?\d*\.\d+|\d+)'))
                result[image_idx] += 1
    num = 0
    for i in range(2733):
        print("# {} is {}".format(i, result[i]))      
    print(num)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='log.txt')
    args = parser.parse_args()
    abspath = os.path.join(os.getcwd(), args.log)
    extract(abspath)
