# IMDB movie reviews - sentiment analysis
Project created for *Text Mining and Social Media Mining* classes at WNE UW

Language:
 * English - classes, notebook

Semester: III (MA studies)

## About
The main objective of this project was to solve some NLP task, so we chose sentiment analysis with Neural Networks (CNNs and RNNs). The big part of this project was to prepare data, firstly we checked for nulls and balance (50/50 => accuracy as a metric), performed entry level cleaning (regex) and label encoding. Our Networks was based on pretrained word embeddings (GloVe) so we had to upload it, build own vocabulary and check embedding coverage what showed the need of more detailed cleaning (regex). After that we added constraint on max review length (there were some outliers on that field). We splitted our data to train, dev and test set (by hand). EDA was finished by graphs - words counts and clouds. To model our data with word embeddings we had to tokenize our data (by hand). Next we proposed 1D convolutional neural network which achived 86% accuracy on the test set. After that we performed confusion matrix analysis and looked at some misclassified examples. We compare our CNN with VADER model which obtained approx 70% accuracy.
At the end we performed some experimets - CNNs with different maximum length of the review and different GloVe dimensions. We tried Bidirectional RNN, but the training was pretty long (we had no more time) and results were poor so we interrupted kernel on 3/4 epoch. We checked the results when stoplist applied (worse).

Findings
 - 100D GloVe is enough for this task
 - 500 maximum length of the review gave the best results
 - CNNs outperforms RNN and VADER
 - negative reviews are completely different even on high level look (most frequent words, words clouds)

In this project I learnt a lot about NLP. I found data preparation very cool, especially hand coded tokenization and sentence vectorization. All the vocabulary, coverage and cleaning with regex stuff was very interesting for me. I found visualizations in NLP tasks very cool too. It was one of my favourite problem during my studies. 

## Repository description
 - Kuzma_Odziemczyk_project.ipynb - Jupyter Notebook with analysis (Python)
 - own.py - some functions used in notebook exported for easier experiments

## Technologies
 - Python (pandas, numpy, matplotlib.plt, seaborn, regex, scikit-learn, keras)
 - Jupyer Notebook

## Authors
 - Bart??omiej Ku??ma [(GitHub)](https://github.com/bartekkuzma?fbclid=IwAR3ECTUCuv44NIAUiOLB8H-LDyHMCyUPCQr86Z4KgH2-iaOHVSz6uatXZQY)
 - Maciej Odziemczyk

