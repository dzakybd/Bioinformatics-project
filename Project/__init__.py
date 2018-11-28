import os
import logging as logger

logger.basicConfig(level=logger.INFO, format='> %(message)s')

cancer_types = ['BRCA', 'BLCA', 'PRAD', 'LUSC', 'THCA', 'LUAD', 'HNSC']
data_location = os.path.abspath(os.path.join(os.path.dirname(__file__)))
train_path = os.path.join(data_location, 'pickled_data', 'training_data')
test_path = os.path.join(data_location, 'pickled_data', 'testing_data')
