# Prediction of Cancer Risk using DNA Methylation Imbalance Data
UCLA Masters Capstone Project

## Background
**Cancer** is the most common human genetic disease and the problem of classifying subjects/patients into disease categories (tumor/normal) is of common occurrence in medical research. Over the past years rapid advances in next-generation sequencing technology has led to the timely advent of **The Cancer Genome Atlas (TCGA)** project which provides the most comprehensive genomic data for various kinds of cancer. The project publishes both clinical (recurrence, survival & treatment resistance) and molecular profiles (Exon (mRNA) expression, DNA methylation, Copy Number Variations (CNV) & Single Nucleotide Polymorphism (SNP)) for both tumor samples and normal controls. Data from TCGA  has improved the ability of oncologists to diagnose, treat, and prevent cancer through a better understanding of the genetic basis of this disease. Previous research indicates classification of patient samples using TCGA Exon expression or SNP datasets. However, recent study show that DNA methylation act as better bio-markers and help in improving the dichotomous outcome (tumor/normal) based on several features. At the same time TCGA data is faced with the problem of class imbalance and high dimensionality of data leading to an increase in the false negative rate. The project aims to tackle these problem and simultaneously lead to the development of a new approach for prognosis of different kinds of cancer using DNA methylation data.

## Approach
The project uses the **Synthetic Minority Oversampling Technique (SMOTE)** algorithm in the pre-processing phase as a method to maintain a balanced class distribution. SMOTE is combined with the **T-Link under-sampling technique** for data cleaning in order to remove noise. To reduce the feature space of the data only those genes for which mutations have been causally implicated in cancer are considered, these are obtained through resources like **The Cancer Gene Census (COSMIC) and Clinical Interpretation of Variants in Cancer (CIVic)**. Classification of patient samples is then performed utilizing several machine learning algorithms of Logistic Regression, Random Forest and Gaussian Naive Bayes. Each classifier performance is evaluated using appropriate performance measures. The methodology is applied on the TCGA DNA Methylation data for 7 various cancer types.

# Computational Genomics Final Project #

In this project, we are proposing a classification engine using somatic single-nucleotide polymorphisms (SNPs) to predict patients who are significantly at risk of developing HGPCa. 

For a detailed understanding of the study, see the writeup attached to this repo.

## Running Instructions ##
```
data_directory='/data'
results_directory='/results'
sh process_data.sh $data_directory $results_directory
sh train_model.sh $data_directory $results_directory
sh test_model.sh $data_directory $results_directory
```
## Things To Do ##
- [X] Setup data pipeline
- [X] Parse gleason labels
- [X] Tie gleason labels to entity IDs in .maf file
- [X] Parse SNP data
- [X] Tie SNP data to gleason labels
- [X] Implement cross validation
- [X] Use Sci-kit for 3 models and evaluate on cross-validation data
- [X] Evaluate performance of different methods 
- [X] Write final paper
- [X] Write powerpoint

