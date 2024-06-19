import json, os, random, io, pdb
from tqdm import tqdm
from random import random
import numpy as np
import glob
import pickle as pkl
import re
test_loc = '/data/coding/apps'
problems = glob.glob(test_loc + '*')
problems = sorted(problems) # Pin some ordering
print(problems)