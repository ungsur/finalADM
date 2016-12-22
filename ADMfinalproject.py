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
    print(i)
    print(doc_outcome[i][0])
    texts.append((doc_outcome[i][0],stemmed_tokens))
    i=i+1

print("doc_set done!")
os.system('say "doc set done"')
model_dict={}
def create_learning_set(text_arr,set_size,model_name):
    count_1 = 0
    count_2 = 0
    print(len(text_arr))
    temp_outcome = []
    for i in range(len(text_arr)):
        print(count_1)
        print(text_arr[i][0])
        if text_arr[i][0] == 1 and count_1 < set_size/2:
            temp_outcome.append((text_arr[i][0],text_arr[i][1]))
            count_1 = count_1 + 1
        elif text_arr[i][0] == -1 and count_2 < set_size/2:
            temp_outcome.append((text_arr[i][0],text_arr[i][1]))
            count_2 = count_2 + 1
    return(temp_outcome)

texts_outcome_p2 = create_learning_set(texts,500,"lda_model_corpus_p2")
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

corpus_p2 = make_corpus(texts_outcome_p2)
#corpus_p = make_corpus(texts_learn_p)
#corpus_2p = make_corpus(texts_learn_2p)
#corpus_all = make_corpus(texts_learn_all)

def make_model_and_corpus(texts_tuple,num_top, num_pass, model_name):
    text_list = [x[1] for x in texts_tuple]
    temp_corpus = [dictionary.doc2bow(text) for text in text_list]
    print(model_name)
    temp_model=models.ldamodel.LdaModel(corpus=temp_corpus,num_topics=num_top,id2word=dictionary, passes=num_pass)
    temp_model_corpus = temp_model[temp_corpus]
    temp_outcomes = [x[0] for x in texts_tuple]
    model_dict[model_name] = {"model": temp_model,"model_corpus": temp_model_corpus, "outcomes":temp_outcomes}
    print(model_dict)
    #model_dict[model_name] = {"model": models.ldamodel.LdaModel(corpus_name, num_topics=num_top, id2word = dictionary, passes=pass_num)}
    
       #modelname.save("/Users/rungsunan/spyder/yelpproject/" + str(modelname))
      # print(modelname + " for " + corpus + " complete!")
       
make_model_and_corpus(texts_outcome_p2,2,5,"lda_model_p2")
#make_model_and_corpus(corpus_p,2,5,"lda_model_corpus_p")
#make_model_and_corpus(corpus_2p,2,5,"lda_model_corpus_2p")
#make_model_and_corpus(corpus_all,2,5,"lda_model_corpus_all")
def save_model_and_corpus(modelname):
    model_dict[modelname]['model'].save('/Users/rungsunan/'+ modelname + '.model')
    corpora.SvmLightCorpus.serialize('/Users/rungsunan/'+ modelname + '.svmlight', model_dict[modelname]['model_corpus'],labels=model_dict[modelname]['outcomes'])
save_model_and_corpus('lda_model_corpus_p2')

def load_ldamodel(modelname):
    load_svmlight_file("/Users/rungsunan/" + modelname + "_corpus.svmlight")

