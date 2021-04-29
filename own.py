import time
import re
import pandas as pd
import numpy as np
from keras.layers import Embedding
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_vocabulary(data, return_dataframe = False):
    '''
    function creates vocabulary for given series of sentences
    data - pd.Series with with text data
    return_dataframe - return sorted df if True

    return vocabulary (dictionary, keys: words, values: counts)
    '''
    vocabulary = {}

    for sentence in data:
        words = sentence.split()
        for word in words:
            if word not in vocabulary.keys():
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    if return_dataframe:
        vocabulary = pd.DataFrame(vocabulary.items(), columns = ["word", "count"]).sort_values("count", ascending = False)

    return vocabulary


def coverage(data, embedding, info = False):
    '''
    function to check coverage with given embedding
    data - pd.Series with text data
    embedding - word embedding dictionary
    info - coverage percentage printed if true

    return covered, non_covered (dicts key: word, value: count)
    '''

    vocabulary = get_vocabulary(data)

    covered = {}
    non_covered = {}

    for word in vocabulary.keys():
        if word in embedding.keys():
            covered[word] = vocabulary[word]
        else:
            non_covered[word] = vocabulary[word]

    covered = pd.DataFrame(covered.items(), columns = ["word", "count"]).sort_values("count", ascending = False)
    non_covered = pd.DataFrame(non_covered.items(), columns = ["word", "count"]).sort_values("count", ascending = False)

    if info:
        print("Covered words percentage: {}% raw: {}%".format(round(
            covered["count"].sum()/(covered["count"].sum()+non_covered["count"].sum()),
            4), round(covered.shape[0]/(covered.shape[0]+non_covered.shape[0]), 4)))
        print("Non-covered words percentage: {}% raw: {}%".format(round(
            non_covered["count"].sum()/(covered["count"].sum()+non_covered["count"].sum()),
            4), round(non_covered.shape[0]/(covered.shape[0]+non_covered.shape[0]), 4)))


    return covered, non_covered


def clean(x):
    x = re.sub("it's", 'it is', x)
    x = re.sub("don't", 'do not', x)
    x = re.sub("i'm", 'i am', x)
    x = re.sub("doesn't", 'does not', x)
    x = re.sub("didn't", 'did not', x)
    x = re.sub("can't", 'can not', x)
    x = re.sub("that's", 'that is', x)
    x = re.sub("i've", 'i have', x)
    x = re.sub("isn't", 'is not', x)
    x = re.sub("there's", 'there is', x)
    x = re.sub("he's", 'he is', x)
    x = re.sub("wasn't", 'was not', x)
    x = re.sub("you're", 'you are', x)
    x = re.sub("couldn't", 'could not', x)
    x = re.sub("you'll", 'you will', x)
    x = re.sub("she's", 'she is', x)
    x = re.sub("i'd", 'i would', x) # assumption that i'd means i wolud, not i had (simplification)
    x = re.sub("they're", 'they are', x)
    x = re.sub("won't", 'will not', x)
    x = re.sub("wouldn't", 'would not', x)
    x = re.sub("i'll", 'i will', x)
    x = re.sub("aren't", 'are not', x)
    x = re.sub("haven't", 'have not', x)
    x = re.sub("what's", 'what is', x)
    x = re.sub("you've", 'you have', x)
    x = re.sub("who's", 'who is', x)
    x = re.sub("let's", 'let us', x)
    x = re.sub("'the", 'the', x)
    x = re.sub("we're", 'we are', x)
    x = re.sub("weren't", 'were not', x)
    x = re.sub("you'd", 'you would', x) # assumption that you'd means you wolud, not you had (simplification)
    x = re.sub("hasn't", 'has not', x)
    x = re.sub("shouldn't", 'should not', x)
    x = re.sub("here's", 'here is', x)
    x = re.sub("hadn't", 'had not', x)
    x = re.sub("we've", 'we have', x)
    x = re.sub("they've", 'whey have', x)
    x = re.sub("ain't", 'not', x) # simplification
    x = re.sub("would've", 'would have', x)
    x = re.sub("could've", 'could have', x)
    x = re.sub("he'd", 'he would', x) # assumption that he'd means he wolud, not he had (simplification)
    x = re.sub("we'll", 'we will', x)
    x = re.sub("they'd", 'they would', x) # assumption that they'd means they wolud, not they had (simplification)
    x = re.sub("it'll", 'it will', x)
    x = re.sub("they'll", 'they will', x)
    x = re.sub("he'll", 'he will', x)
    x = re.sub("we'd", 'we would', x) # assumption that we'd means we wolud, not we had (simplification)
    x = re.sub("should've", 'should have', x)
    x = re.sub("she'd", 'she would', x) # assumption that she'd means she wolud, not she had (simplification)
    x = re.sub("she'll", 'she will', x)
    x = re.sub("'i", 'i', x)
    x = re.sub("who've", 'who have', x)
    x = re.sub("it'd", 'it would', x) # assumption that she'd means she wolud, not she had (simplification)
    x = re.sub("bandw", 'b and w', x)
    x = re.sub("that'll", 'that will', x)
    x = re.sub("imho", 'in my humble opinion', x)
    x = re.sub("who'd", 'who would', x) # assumption that who'd means who wolud, not who had (simplification)
    x = re.sub("o'", 'o', x)
    x = re.sub("must've", 'must have', x)
    x = re.sub("'what", 'what', x)
    x = re.sub("it'", 'it', x)
    x = re.sub("its'", 'its', x)

    return x

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    # number of training examples
    m = X.shape[0]

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len)) # carry out padding

    # loop over training examples
    for i in range(m):

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            if w in word_to_index.keys():
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
            else:
                X_indices[i, j] = word_to_index["<OOV>"]

            # Increment j to j + 1
            j = j + 1

    #X_indices = np.r_[np.zeros((1,maxLen)), X_indices]
    return X_indices

def results(real, pred):
    cm=confusion_matrix(real, pred)
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))

    sns.set(font_scale=1) # for label size
    sns.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 11}) # font size

    plt.show()
    print(classification_report(real, pred))


def plots(model, epoch, version="both"):
    plt.style.use("seaborn")
    plt.figure()
    plt.rcParams["font.family"] = "Courier New"

    if version == "both":
        plt.plot(np.arange(0, epoch), model.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epoch), model.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epoch), model.history["acc"], label="train_acc")
        plt.plot(np.arange(0, epoch), model.history["val_acc"], label="val_acc")
        plt.title("Loss and Accuracy during training")
        plt.ylabel("Loss/Accuracy")

    elif version == "loss":
        plt.plot(np.arange(0, epoch), model.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epoch), model.history["val_loss"], label="val_loss")
        plt.title("Loss during training")
        plt.ylabel("Loss")

    elif version == "acc":
        plt.plot(np.arange(0, epoch), model.history["acc"], label="train_acc")
        plt.plot(np.arange(0, epoch), model.history["val_acc"], label="val_acc")
        plt.title("Accuracy during training")
        plt.ylabel("Accuracy")

    plt.xlabel("Epoch #")
    plt.legend(loc="lower left")
    plt.show()

def example(i, misclass):
    if i<len(misclass):
        print(f"True value: {Y_test[misclass[0]]}\nPredicted probability {preds[misclass[0]][0]}\nPredicted value: {preds_value[misclass[0]]}\nThe review:\n{X_test[misclass[0]]}")
    else:
        print("Try lower number")


def dataset(length, embedding, stopwords=False, train_part=0.7, dev_part=0.2, test_part=0.1, ran_st=2021):
    print(f"Max sentence length: {length}\nEmbedding dimension: {embedding}D\nStopwords used: {stopwords}\nTrain part: {train_part}\nDev part: {dev_part}\nTest part: {test_part}\nRandom state: {ran_st}")
    startFull = time.time()
    #wczytane danych
    data = pd.read_csv("IMDB Dataset.csv")

    # entry level data clearing
    # strategy is to substitute with space then remove unnecessary spaces

    # remove <> and all between
    data.review = data.review.apply(lambda x: re.sub("<.*?>", " ", x))
    # substitute "&" with "and"
    data.review = data.review.apply(lambda x: re.sub("[&]", "and", x))
    #remove punctuation
    data.review = data.review.apply(lambda x: re.sub("[^a-zA-Z0-9 ']", " ", x))
    # remove numbers >10
    data.review = data.review.apply(lambda x: re.sub("([2-9][0-9]|1[1-9]|\d{3,})", " ", x))
    # delete unnecessary spaces
    data.review = data.review.apply(lambda x: re.sub(" +", " ", x))
    # transform to lowercase
    data.review = data.review.apply(lambda x: x.lower())

    # label encoding
    data.sentiment = data.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    # train/dev/test split
    data = data.sample(frac = 1, random_state = ran_st)
    train_size = train_part
    dev_size = dev_part
    test_size = test_part

    train = data.iloc[range(int(data.shape[0]*train_size)), :].copy()
    X_train = train.review.values
    Y_train = train.sentiment.values

    dev = data.iloc[range(train.shape[0], train.shape[0]+int(data.shape[0]*dev_size)), :].copy()
    X_dev = dev.review.values
    Y_dev = dev.sentiment.values

    test = data.iloc[range(train.shape[0]+dev.shape[0], data.shape[0]), :].copy()
    X_test = test.review.values
    Y_test = test.sentiment.values


    embeddings_index = {}
    f = open(f"glove.6B.{embedding}d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    #print('Found %s word vectors.' % len(embeddings_index))

    # look for different-shaped embeddings

    # inicialize dict (key: word, value: embedding shape)
    words = {}
    # position, to check if there is a pattern
    iis = []

    # loop
    for i, word in enumerate(embeddings_index.keys()):
        # current word embedding shape
        shape = embeddings_index[word].shape[0]
        # condtition check
        if shape != embedding:
            # saving
            words[word] = shape
            iis.append(i)
    # unknown token computation
    #start = time.time()
    unknown = sum(embeddings_index.values())/len(embeddings_index)
    #stop = time.time()
    #print("Token execution time: {} seconds".format(round(stop-start, 2)))
    # saving computed unknown to embedding
    embeddings_index["<OOV>"] = unknown

    #detailed regex
    X_train = pd.Series(X_train).apply(lambda x: clean(x)).values

    X_dev = pd.Series(X_dev).apply(lambda x: clean(x)).values
    X_test = pd.Series(X_test).apply(lambda x: clean(x)).values

    # removing too long reviews
    selected_max_length = length
    temp_df = pd.DataFrame(np.r_[X_train,X_dev,X_test], columns=["review"])
    temp_df["length"] = temp_df.review.apply(lambda x: len(x.split()))
    temp_df = temp_df.where(temp_df.length <= selected_max_length).dropna().reset_index(drop = True)
    #print("data shape:", temp_df.shape)
    maxLen = selected_max_length

    temp_train = pd.DataFrame(np.c_[X_train,Y_train], columns = ["review", "sentiment"])
    temp_train["length"] = temp_train.review.apply(lambda x: len(x.split()))
    temp_train = temp_train.where(temp_train.length <= selected_max_length).dropna().reset_index(drop = True)
    X_train = temp_train.review.values
    Y_train = temp_train.sentiment.values.astype(int)

    temp_dev = pd.DataFrame(np.c_[X_dev,Y_dev], columns = ["review", "sentiment"])
    temp_dev["length"] = temp_dev.review.apply(lambda x: len(x.split()))
    temp_dev = temp_dev.where(temp_dev.length <= selected_max_length).dropna().reset_index(drop = True)
    X_dev = temp_dev.review.values
    Y_dev = temp_dev.sentiment.values.astype(int)

    temp_test = pd.DataFrame(np.c_[X_test,Y_test], columns = ["review", "sentiment"])
    temp_test["length"] = temp_test.review.apply(lambda x: len(x.split()))
    temp_test = temp_test.where(temp_test.length <= selected_max_length).dropna().reset_index(drop = True)
    X_test = temp_test.review.values
    Y_test = temp_test.sentiment.values.astype(int)

    # get training vocabulary
    # stop words exclusion
    if stopwords==True:
        train_vocab = list(get_vocabulary(X_train).keys())
        train_vocab = list(set(train_vocab)-(STOPWORDS))
    elif stopwords==False:
        train_vocab = list(get_vocabulary(X_train).keys())


    # build a token dictionary
    word_index = {}
    words_in_embedding = embeddings_index.keys()
    i = 0
    for word in train_vocab:
        if word in words_in_embedding:
            word_index[word] = i
            i += 1
        else:
            if not word_index.get("<OOV>"):
                word_index["<OOV>"] = i
                i += 1

    X_train_ind = sentences_to_indices(X_train, word_index, maxLen)
    X_dev_ind = sentences_to_indices(X_dev, word_index, maxLen)

    embedding_dim = embedding
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=maxLen,
                                trainable=False)

    stopFull = time.time()
    print("Total execution time: {} seconds".format(round(stopFull-startFull, 2)))

    return embedding_layer, maxLen, word_index, X_train_ind, Y_train, X_dev_ind, Y_dev, X_test, Y_test
