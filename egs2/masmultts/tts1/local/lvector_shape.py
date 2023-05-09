#!/usr/bin/env python3
#  2022, The University of Tokyo; Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import pathlib
import subprocess

import kaldiio
import numpy as np
import torch
from tqdm.contrib import tqdm
import git

from espnet2.fileio.sound_scp import SoundScpReader

import lang2vec.lang2vec as l2v

def main():
    lcode = "afr"
    items = ["fam", "syntax_average", "phonology_average", "inventory_average"]
    for item in items:
        out = l2v.get_features(lcode, item)
        vec = np.asarray(out[lcode])
        print(f"{item}: {vec.shape}")

if __name__ == "__main__":
    main()