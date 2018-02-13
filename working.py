import zipfile
import collections
import operator
import random
import numpy as np

''' Global variable declaration '''

f1 = "text8.zip"
window_size = 1
size = 253855 #no of most common words
epoch = 4
lr = 0.1
inputlayer_neurons = size
hiddenlayer_neurons = 100
output_neurons = size
word_index_dict = {}
count = [['UNK', -1]]
table_size = 1e7
words_p_list = []
no_negative_samples = 2

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))

''' Global variable declaration ends '''

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def read_data(filename):
    ''' Returns list of all words from zipfile '''
    with zipfile.ZipFile(filename) as f:
        m = f.read(f.namelist()[0]).decode('UTF-8').split()
        return m

def build_set(words, n_words):
    global word_index_dict, count
    count.extend(collections.Counter(words).most_common(n_words - 1))
    word_index_dict['UNK']=0
    for var in range(0,n_words-1):
        word_index_dict[count[var][0]] = var+1
    print("Built dictionary!")

    dictionary = dict()
    ''' Create dict of form (word : i) i is incremented '''
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
    count[0][1] = unk_count
    #print("Count's length is: ", len(count))


def generate_p_negative_samples():
    global count
    s1 = 0
    p_list = []
    for var in count:
        s1+=var[1]**0.75
    for var in count:
        p_list.append((var[1]**0.75)/s1)
    return p_list

def generate_words_negative_samples(prob_list):
    global words_p_list
    added = 0
    table_size = 1e7
    i = 0
    while(i<len(prob_list)):
        added = 0
        while(added/table_size)<prob_list[i]:
            words_p_list.append(count[i][0])
            added += 1
        i+=1

def generate_word_vec(word):
    global size
    try:
        index = word_index_dict[word]
    except:
        index = 0
    vec = list()
    vec = [0 for _ in range(size)]
    vec[index] = 1
    return index, vec

def gen_ind(word):
    try:
        index = word_index_dict[word]
    except:
        index = 0
    return index

def main():
    vocabulary = read_data(f1) #list of tokenized words from corpus
    len_vocabulary = len(vocabulary)
    print(len_vocabulary, "is the length of the vocabulary")
    # 17005207 total no of words,  253855 unique words
    build_set(vocabulary,size)
    p=generate_p_negative_samples()
    generate_words_negative_samples(p)
    #print("Size is ",size)
    #print("Word_index_dict is ",len(word_index_dict))
    #print("Prob list is",len(p))
    #print("Count is ", len(count))
    print("Training begins!")
    for vvv in range(4):
        print("Running for epoch number :", vvv)

        for j in range(len_vocabulary-10):
            #print("Length of vocabulary is ", len_vocabulary,"and run is",j)
            i = random.randint(1,(len_vocabulary-10))
            word = vocabulary[random.randint(1,(len_vocabulary-10))]
            i1 = gen_ind(word)
            ops = []
            ops_indices = []
            ops_words = []
            for w_i in range(1,window_size):
                #ops.append(generate_word_vec(vocabulary[i+w_i]))
                ops_words.append(vocabulary[i+w_i])
                try:
                    ops_indices.append(word_index_dict[vocabulary[i+w_i]])
                except:
                    ops_indices.append(0)
            
                #ops.append(generate_word_vec(vocabulary[i-w_i]))
                ops_words.append(vocabulary[i+w_i])
                try:
                    ops_indices.append(word_index_dict[vocabulary[i-w_i]])
                except:
                    ops_indices.append(0)
            train(ops, i1, ops_indices, ops_words)

        print("Done training")
    
    with open('vocab.txt','r') as f:
        data = f.readlines()

    f2 = open('vec_2_neg_pls.txt','a')
    for var in data:
        w1 = var.replace('\n','')
        f2.write(w1)
        f2.write(" ")
        try:
            index1 = word_index_dict[w1]
            #print("Index is: ",index1)
        except:
            index1 = 0
        #print("Index is: ", index1)
        np.savetxt(f2, wh[index1], newline = " ",fmt = '%f')
        f2.write("\r\n")

def train(outputs, ip_index, op_indices, op_words):
    global lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons, wh, wout, words_p_list
    #y = np.array(outputs)
    no_of_trials = len(op_indices)
    #X = np.array([x])
    ip = ip_index

    for j in range(no_of_trials):
        neg_sum = 0
        neg_s_words = []
        y1 = op_indices[j]
        neg_s_words.append(op_words[j])

        for v1 in range(no_negative_samples):
            neg_s_words.append(words_p_list[random.randint(0,1e7)])

        for var in neg_s_words:
            #xx, y_g = generate_word_vec(var)
            var1 = y1
            try:
                yindex = word_index_dict[var]
            except:
                yindex = 0
            if(yindex==y1):
                yy = 1
            else:
                yy = 0
            neg_sum += (EIj(ip,var1,yy)*wout.T[var1])
            wout.T[var1] -= lr*EIj(ip,var1,yy)*wh[ip]
        wh[ip] -= lr*neg_sum


def EIj(ip,j,t):
    return (sigmoid(wout.T[j].dot(wh[ip].T)) - t)

if __name__ == "__main__":
    main()


