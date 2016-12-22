# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import os
from gensim import corpora, models
import json
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.datasets import load_svmlight_file

from sklearn import  svm
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import pandas 
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier

#Importing business json file to dictionary with all businesses and categories 
bus_list = []
whole_dict = {}
res_dict = {}
smaller_dict = {}
for line in open('/Users/rungsunan/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json')    :
    temp_dict = {}    
    temp_dict.update(json.loads(line))
    busid = temp_dict["business_id"]
    categories = temp_dict["categories"]
    whole_dict[busid] = categories
    for i in range(len(categories)):
        if categories[i]=="Restaurants":
            res_dict[busid]=categories[i]
        if categories[i]=="Chinese" or categories[i]=="Italian" :
            smaller_dict[busid]={"categories":categories[i]}

print("size of whole busid:categorylist: " + str(len(whole_dict)))  
print("number of restaurants: " + str(len(res_dict)))
print("number of chinese or italian restaurants: " + str(len(smaller_dict)))


review_dict = {}
outcome_list = []
for line in open('/Users/rungsunan/Downloads/yelp_dataset_challenge_academic_dataset/500000_yelp_academic_dataset_review.json'):
    temp_dict = {}    
    temp_dict.update(json.loads(line))
    review_business = temp_dict["business_id"]
    for business in smaller_dict.keys():
        if review_business == business:
            if review_business in review_dict:
                review_dict[review_business]["categories"] = smaller_dict[business]["categories"]
                review_dict[review_business]["revtext"].append(temp_dict["text"])
                review_dict[review_business]["rating"].append(temp_dict["stars"])
            else:
                review_dict[review_business]={"categories":smaller_dict[business]["categories"],"revtext":[temp_dict["text"]],"rating":[temp_dict["stars"]]}
print("number of businesses in review document: " + str(len(review_dict)))

tokenizer = RegexpTokenizer(r'\s+', gaps=True)

# create English stop words list
en_stop = get_stop_words('en')
# print(en_stop)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

doc_set = []
outcomes_list = []
doc_outcome = []
for k, v in review_dict.items():
    for i in range(len(review_dict[k]["revtext"])):
        if review_dict[k]["categories"] == "Chinese":
            outcomes_list.append(-1)
            doc_outcome.append((-1,review_dict[k]["revtext"][i]))
        else:
            outcomes_list.append(1)
            doc_outcome.append((1,review_dict[k]["revtext"][i]))

# list for tokenized documents in loop
texts = []
punctuation_string = '"?!@#$%^&*()\';:+,\.-'

# loop through document list
for i in range(len(doc_outcome)):
    # clean and tokenize document string
    raw = doc_outcome[i][1].lower()
   
    for l in range(len(punctuation_string)):
        raw = raw.replace(punctuation_string[l], '')
            
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [j for j in tokens if not j in en_stop]
    # print(tokens)
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(k) for k in stopped_tokens]
    
    # add tokens to list
    texts.append((doc_outcome[i][0],stemmed_tokens))
    i=i+1

print("doc_set done!")
os.system('say "doc set done"')
model_dict={}
def create_learning_set(text_arr,set_size,model_name):
    count_1 = 0
    count_2 = 0
    temp_outcome = []
    for i in range(len(text_arr)):
        if text_arr[i][0] == 1 and count_1 < set_size/2:
            temp_outcome.append((text_arr[i][0],text_arr[i][1]))
            count_1 = count_1 + 1
        elif text_arr[i][0] == -1 and count_2 < set_size/2:
            temp_outcome.append((text_arr[i][0],text_arr[i][1]))
            count_2 = count_2 + 1
    return(temp_outcome)

#outcomes_list_p, texts_learn_p = create_learning_set(texts,1000)
#outcomes_list_2p, texts_learn_2p = create_learning_set(texts,2000)
#outcomes_list_all, texts_learn_all = create_learning_set(texts,len(texts))

dictionary = corpora.Dictionary([x[1] for x in texts])
print("there are: " + str(len(texts)) + " review documents in the dictionary")
print("dictionary done!")
os.system('say "Finished dictionary"')

def make_corpus(texts_tuple):
    text_list = [x[1] for x in texts_tuple]
    return  [dictionary.doc2bow(text) for text in text_list]


#corpus_p = make_corpus(texts_learn_p)
#corpus_2p = make_corpus(texts_learn_2p)
#corpus_all = make_corpus(texts_learn_all)

def make_model_and_corpus(texts_tuple,num_top, num_pass, model_name):
    text_list = [x[1] for x in texts_tuple]
    temp_corpus = [dictionary.doc2bow(text) for text in text_list]
    print(model_name)
    temp_model=models.LdaMulticore(corpus=temp_corpus,num_topics=num_top,id2word=dictionary, passes=num_pass)
    temp_model_corpus = temp_model[temp_corpus]
    temp_outcomes = [x[0] for x in texts_tuple]
    model_dict[model_name] = {"model": temp_model,"model_corpus": temp_model_corpus, "outcomes":temp_outcomes}
    print(model_dict)
    #model_dict[model_name] = {"model": models.ldamodel.LdaModel(corpus_name, num_topics=num_top, id2word = dictionary, passes=pass_num)}
    
       #modelname.save("/Users/rungsunan/spyder/yelpproject/" + str(modelname))
      # print(modelname + " for " + corpus + " complete!")
       
#make_model_and_corpus(corpus_p,2,5,"lda_model_corpus_p")
#make_model_and_corpus(corpus_2p,2,5,"lda_model_corpus_2p")
#make_model_and_corpus(corpus_all,2,5,"lda_model_corpus_all")
def save_model_and_corpus(modelname):
    model_dict[modelname]['model'].save('/Users/rungsunan/'+ modelname + '.model')
    corpora.SvmLightCorpus.serialize('/Users/rungsunan/'+ modelname + '_corpus.svmlight', model_dict[modelname]['model_corpus'],labels=model_dict[modelname]['outcomes'])


def load_ldamodel(modelname):
    print(modelname)
    X_temp, y_temp = load_svmlight_file("/Users/rungsunan/" + modelname)
    return (X_temp,y_temp)


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    return clf

def histo_corpus(modelname):
    histo = []
    corpus = model_dict[modelname]['model_corpus']
    for doc in corpus:
        for i in range(len(doc)):
            if doc[i][1] > .1:
                histo.append(doc[i][0])
    return histo


def best_topics (texts_tuple):
    grid = defaultdict(list)
    param_list = []
    perplex_list = []
    perword_list = []
    text_list = [x[1] for x in texts_tuple]
    temp_corpus = [dictionary.doc2bow(text) for text in text_list]
    number_of_words = sum(cnt for document in temp_corpus for _, cnt in document)
    parameter_list = [2,3,4,5,10,15]
    for parameter_value in parameter_list:      
        print ("starting pass for parameter_value = %.3f" % parameter_value)
        model = models.LdaMulticore(corpus=temp_corpus, workers=None, id2word=dictionary, num_topics=parameter_value, iterations=20)
        perplex = model.bound(temp_corpus) # this is model perplexity not the per word perplexity
        print ("Total Perplexity: %s" % perplex)
        param_list.append(parameter_value)
        perplex_list.append(perplex)
        grid[parameter_value].append(perplex)

    
        per_word_perplex = np.exp2(-perplex / number_of_words)
        perword_list.append(per_word_perplex)
        print ("Per-word Perplexity: %s" % per_word_perplex)
        grid[parameter_value].append(per_word_perplex)
        #model.save(data_path + 'ldaMulticore_i10_T' + str(parameter_value) + '_training_corpus.lda')
        print

    for numtopics in parameter_list:
        print (numtopics, '\t',  grid[numtopics])
    df = pandas.DataFrame(grid)
    ax = plt.figure(figsize=(7, 4), dpi=300).add_subplot(111)
    df.iloc[1].transpose().plot(ax=ax,  color="#254F09")
    plt.xlim(parameter_list[0], parameter_list[-1])
    plt.ylabel('Perplexity')
    plt.xlabel('topics')
    plt.title('')
    plt.show()
    #df.to_pickle(data_path + 'gensim_multicore_i10_topic_perplexity.df')

    
texts_outcome_p2 = create_learning_set(texts,1000,"lda_model_corpus_p2")
texts_outcome_p = create_learning_set(texts,3000,"lda_model_corpus_p")
texts_outcome_2p = create_learning_set(texts,7000,"lda_model_corpus_2p")
texts_outcome_p10k = create_learning_set(texts,10000,"lda_model_corpus_2p")
texts_outcome_all = create_learning_set(texts,40000,"lda_model_corpus_all")

best_topics(texts_outcome_all)
best_topics(texts_outcome_p2)
best_topics(texts_outcome_p)
best_topics(texts_outcome_2p)
best_topics(texts_outcome_p10k)

corpus_all = make_corpus(texts_outcome_all)
corpus_p10k = make_corpus(texts_outcome_p10k)
corpus_p2 = make_corpus(texts_outcome_p2)
make_model_and_corpus(texts_outcome_all,2  ,20,"lda_model_all_3topics")
make_model_and_corpus(texts_outcome_p10k,2  ,20,"lda_model_p10k_2topics")
make_model_and_corpus(texts_outcome_p10k,3  ,20,"lda_model_p10k_3topics")
make_model_and_corpus(texts_outcome_p10k,10  ,20,"lda_model_p10k_10topics")

save_model_and_corpus('lda_model_p10k_2topics')
save_model_and_corpus('lda_model_p10k_3topics')
save_model_and_corpus('lda_model_p10k_10topics')

(X_p10k_2topics, y_p10k_2topics) = load_ldamodel("lda_model_p10k_2topics_corpus.svmlight")
(X_p10k_3topics, y_p10k_3topics) = load_ldamodel("lda_model_p10k_3topics_corpus.svmlight")
(X_p10k_10topics, y_p10k_10topics) = load_ldamodel("lda_model_p10k_10topics_corpus.svmlight")

X_train_p10k_2topics, X_test_p10k_2topics, y_train_p10k_2topics, y_test_p10k_2topics = train_test_split(
  X_p10k_2topics, y_p10k_2topics, test_size=0.2, random_state=0)
X_train_p10k_3topics, X_test_p10k_3topics, y_train_p10k_3topics, y_test_p10k_3topics = train_test_split(
  X_p10k_3topics, y_p10k_3topics, test_size=0.2, random_state=0)
X_train_p10k_10topics, X_test_p10k_10topics, y_train_p10k_10topics, y_test_p10k_10topics = train_test_split(
  X_p10k_10topics, y_p10k_10topics, test_size=0.2, random_state=0)


k_range = range(1,25)
param_grid = dict(n_neighbors=k_range)
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',n_jobs=-1)
grid.fit(X_p10k_2topics,y_p10k_2topics)
grid.grid_scores_
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
plt.plot(k_range,grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100,1000]}
scores = ['precision','recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score,n_jobs=-1)
    clf.fit(X_train_p10k_2topics, y_train_p10k_2topics)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true_p10k_2topics, y_pred_p10k_2topics = y_test_p10k_2topics, clf.predict(X_test_p10k_2topics)
    print(classification_report(y_true_p10k_2topics, y_pred_p10k_2topics))
    print(confusion_matrix(y_true_p10k_2topics, y_pred_p10k_2topics))
    print()
    os.system('say "doc set done"')

k_range = range(1,25)
param_grid = dict(n_neighbors=k_range)
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',n_jobs=-1)
grid.fit(X_p10k_3topics,y_p10k_3topics)
grid.grid_scores_
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
plt.plot(k_range,grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100,1000]}
scores = ['precision','recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score,n_jobs=-1)
    clf.fit(X_train_p10k_3topics, y_train_p10k_3topics)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true_p10k_3topics, y_pred_p10k_3topics = y_test_p10k_3topics, clf.predict(X_test_p10k_3topics)
    print(classification_report(y_true_p10k_3topics, y_pred_p10k_3topics))
    print(confusion_matrix(y_true_p10k_3topics, y_pred_p10k_3topics))
    print()
    os.system('say "doc set done"')


k_range = range(1,25)
param_grid = dict(n_neighbors=k_range)
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy',n_jobs=-1)
grid.fit(X_p10k_10topics,y_p10k_10topics)
grid.grid_scores_
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
plt.plot(k_range,grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

tuned_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100,1000]}
scores = ['precision','recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score,n_jobs=-1)
    clf.fit(X_train_p10k_10topics, y_train_p10k_10topics)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true_p10k_10topics, y_pred_p10k_10topics = y_test_p10k_10topics, clf.predict(X_test_p10k_10topics)
    print(classification_report(y_true_p10k_10topics, y_pred_p10k_10topics))
    print(confusion_matrix(y_true_p10k_10topics, y_pred_p10k_10topics))
    print()
    os.system('say "doc set done"')
    
#svm_p2 = train_svm(X_p2, y_p2)
#pred_p2 = svm_p2.predict(X_test_p2)
#print(svm_p2.score(X_test_p2, y_test_p2))
#print(confusion_matrix(pred_p2, y_test_p2))

histo_2p = histo_corpus("lda_model_p2")
plt.hist(histo_2p)

