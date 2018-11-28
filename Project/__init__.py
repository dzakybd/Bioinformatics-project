import os
import logging as logger

#Choose "Tumor" or "hgpca"
attribute = "hgpca"

logger.basicConfig(level=logger.INFO, format='> %(message)s')

dataset_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
clinical_path = os.path.join(dataset_location, 'All_CDEs.tsv')
meth_path = os.path.join(dataset_location, 'PRAD.meth.by_mean.data.tsv')

train_test_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_test_data'))
train_path = os.path.join(train_test_location, 'train_data_')
test_path = os.path.join(train_test_location, 'test_data_')

gene_catalog_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gene_catalog'))
cosmic_path = os.path.join(gene_catalog_path, 'cosmic_gene_census.csv')
civic_path = os.path.join(gene_catalog_path, 'civic_gene_summaries.tsv')
