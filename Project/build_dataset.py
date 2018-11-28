import pandas as pd
import seaborn as sns
from __init__ import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter(action='ignore')



# dark grey color
ALMOST_BLACK = '#262626'


def visualize_data(data, attribute):
    """
    Visualize the data in 2D
    :param data: data for visualization
    """

    # Instantiate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    x_vis = pca.fit_transform(data[data.columns[:-1]])
    y = data[attribute].as_matrix()

    # Plot the original data
    # Plot the two classes
    palette = sns.color_palette()

    plt.scatter(x_vis[y == 0, 0], x_vis[y == 0, 1], label="Normal", alpha=0.5,
                edgecolor=ALMOST_BLACK, facecolor=palette[0], linewidth=0.15)
    plt.scatter(x_vis[y == 1, 0], x_vis[y == 1, 1], label=attribute, alpha=0.5,
                edgecolor=ALMOST_BLACK, facecolor=palette[2], linewidth=0.15)

    plt.legend()
    plt.show()


def cosmic_cancer_genes():
    """
    Read COSMIC Cancer Gene Census - catalogue those genes for which mutations have been causally implicated in cancer

    :return: list of cancer genes along with synonyms
    """

    gene_census_data = pd.read_csv(cosmic_path, skipinitialspace=True, usecols=['Gene Symbol', 'Synonyms'])
    gene_census = list(gene_census_data['Gene Symbol'])

    logger.info('Number of Cancer Gene Census: {0}\n'.format(len(gene_census)))

    for synonynm in gene_census_data['Synonyms']:
        if type(synonynm) is str:
            gene_census.extend(synonynm.split(','))

    return gene_census


def civic_cancer_genes():
    """
    Read Clinical Interpretation of Variants in Cancer (Civic) catalogue of cancer genes

    :return: list of cancer genes
    """

    civic_genes_data = pd.read_csv(civic_path, skipinitialspace=True, usecols=['name'], delimiter='\t')
    civic_genes = list(civic_genes_data['name'])

    return civic_genes


def genes_feature_selection(methyl_data, cancer_genes):
    """
    Reduce feature space of protein-binding genes by considering COSMIC & CIVIC data

    :param methyl_data: DNA methylation data with >20000 protein-coding genes
    :return: list of reduced set of genes
    """

    overlap_genes = cancer_genes.intersection(methyl_data.index)

    return methyl_data.ix[overlap_genes]

def add_hgpca_label(methyl_data):
    case_ids = methyl_data.columns.values
    labels = []

    clinical_data = pd.read_csv(clinical_path, skipinitialspace=True, usecols=['bcr_patient_barcode', 'gleason_score'], delimiter='\t')
    clinical_dic = dict(zip(clinical_data['bcr_patient_barcode'], clinical_data['gleason_score']))
    for case_id in case_ids:
        short_id = '-'.join(case_id.split('-')[:-1]).lower()
        gleason_score = clinical_dic.get(short_id)
        if 2 <= gleason_score <= 7:
            labels.append(1)
        elif 8 <= gleason_score <= 10:
            labels.append(0)

    methyl_data.loc[methyl_data.shape[0]] = labels
    methyl_data = methyl_data.rename(index={methyl_data.shape[0] - 1: 'hgpca'})

    return methyl_data

def add_tumor_label(methyl_data):
    """
    Add classification label: tumor '1' | no tumor '0'
    Barcode description - https://wiki.nci.nih.gov/display/TCGA/TCGA+barcode

    :param methyl_data: DNA methylation data
    :return: Data with classification label appended
    """
    case_ids = methyl_data.columns.values
    labels = []

    for case_id in case_ids:
        tumor_type = int(case_id.split('-')[-1])

        if 1 <= tumor_type <= 9:
            labels.append(1)
        elif 10 <= tumor_type <= 19:
            labels.append(0)
        else:
            logger.warning('Tumor Type mismatch for case-id: {0}'.format(case_id))

    methyl_data.loc[methyl_data.shape[0]] = labels
    methyl_data = methyl_data.rename(index={methyl_data.shape[0] - 1: 'Tumor'})

    return methyl_data


def train_test_data(methyl_data, attribute):
    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()

    normal_cases = methyl_data[methyl_data[attribute] == 0]
    logger.info(normal_cases.shape)
    train_normal_cases = normal_cases.sample(frac=0.7, random_state=1)
    logger.info(train_normal_cases.shape)
    test_normal_cases = normal_cases.drop(train_normal_cases.index)
    logger.info(train_normal_cases.shape)

    tumor_cases = methyl_data[methyl_data[attribute] == 1]
    logger.info(tumor_cases.shape)
    train_tumor_cases = tumor_cases.sample(frac=0.7, random_state=1)
    logger.info(train_tumor_cases.shape)
    test_tumor_cases = tumor_cases.drop(train_tumor_cases.index)
    logger.info(test_tumor_cases.shape)

    training_data = training_data.append(train_normal_cases)
    training_data = training_data.append(train_tumor_cases)
    testing_data = testing_data.append(test_normal_cases)
    testing_data = testing_data.append(test_tumor_cases)

    training_data = training_data.sample(frac=1, random_state=1)
    testing_data = testing_data.sample(frac=1, random_state=1)

    logger.info('Pickling training and testing data')
    training_data.to_pickle(train_path+attribute)
    testing_data.to_pickle(test_path+attribute)

    logger.info('Processing completed!')
    visualize_data(training_data, attribute)

    return training_data, testing_data


def build():
    """
    Read and process DNA methylation data of 6 cancer subtypes.
    Processing - feature selection, data imputation & adding classification label (tumor '1' or not tumor '0')

    :return: DNA methylation training and testing dataframe
    """

    logger.info('Process initiated - Building dataset')

    if os.path.isfile(train_path+attribute) and os.path.isfile(test_path+attribute):
        logger.info('Loading pickled data')
        return pd.read_pickle(train_path+attribute), pd.read_pickle(test_path+attribute)

    logger.info('Reading COSMIC Cancer Gene Census')
    gene_census = cosmic_cancer_genes()
    logger.info('Reading CIVIC Cancer Gene Census')
    gene_census.extend(civic_cancer_genes())

    gene_census = set(gene_census)

    logger.info('Reading Methylation data of TCGA PRAD')

    print(meth_path)
    methyl_data = pd.read_csv(meth_path, delimiter='\t', skiprows=[1], index_col=0)

    logger.info('Number of Genes: {0} | Number of Patients: {1}'.format(methyl_data.shape[0], methyl_data.shape[1]))
    logger.info('Preprocessing Methylation data')

    methyl_data = genes_feature_selection(methyl_data, gene_census)

    logger.info('Number of Genes after processing: {0}\n'.format(methyl_data.shape[0]))

    if attribute == "Tumor":
        methyl_data = add_tumor_label(methyl_data)
    elif attribute == "hgpca":
        methyl_data = add_hgpca_label(methyl_data)

    methyl_data = methyl_data.transpose()

    return train_test_data(methyl_data, attribute)

if __name__ == '__main__':
    build()
