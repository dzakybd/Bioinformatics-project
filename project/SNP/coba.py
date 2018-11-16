import numpy as np
import sys
import os 
import pandas as pd

path = "data/snv/TCGA.PRAD.mutect.b97f5495-ab12-48c7-a436-612569242bd7.somatic.maf"
    
#snv_data = np.genfromtxt(path, dtype='str', delimiter='\t')

data = pd.read_csv(path, delimiter = "\t", dtype= str)