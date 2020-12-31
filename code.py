
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import f1_score
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from itertools import chain
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def collect_words(data_path):
    df = pd.read_csv(data_path, dtype=str)
    tokens = set()
    for i in df.iterrows():
        sents  = i[1]['Abstract'].split('$$$')
        sents = ' '.join(sents)
        tokens |= set(word_tokenize(sents))
    return tokens


def get_dataset(data_path, word_dict, n_workers=0):
    """ Load data and return dataset for training and validating.

    Args:
        data_path (str): Path to the data.
    Return:
        output (list of dict): [dict, dict, dict ...]
    """
    dataset = pd.read_csv(data_path, dtype=str)
    formatData = []
    for (idx,data) in dataset.iterrows():
        """
        processed: {
        'Abstract': [[4,5,6],[3,4,2],...]
        'Label': [[0,0,0,1,1,0],[1,0,0,0,1,0],...]
        }
        """
        processed = {}
        processed['Abstract'] = [sentence_to_indices(sent, word_dict) for sent in data['Abstract'].split('$$$')]
        if ('Task 1' in data ):
            processed['Label'] = [label_to_onehot(label) for label in data['Task 1'].split(' ')]
        formatData.append(processed)
    
    return formatData
  
def label_to_onehot(labels):
    """ Convert label to onehot .
        Args:
            labels (string): sentence's labels.
        Return:
            outputs (onehot list): sentence's onehot label.
    """
    label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
    onehot = [0,0,0,0,0,0]
    for l in labels.split('/'):
        onehot[label_dict[l]] = 1
    return onehot
        
def sentence_to_indices(sentence, word_dict):
    """ Convert sentence to its word indices.
    Args:
        sentence (str): One string.
    Return:
        indices (list of int): List of word indices.
    """
    return [word_dict.get(word,UNK_TOKEN) for word in word_tokenize(sentence)] 

class AbstractDataset(Dataset):
  def __init__(self, data, pad_idx, max_len = 500):
        self.data = data
        self.pad_idx = pad_idx
        self.max_len = max_len
  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        return self.data[index]
      
  def collate_fn(self, datas):
        """
        returns:
        Tensor(batch,sentence,words) : input data
        Tensor(batch,sentence,words) : corresponding answer
        list(sentence quantity in each abstract): use in prediction, to remove the redundant sentences (the sentences we padded)
        
        """
        # get max length in this batch
        max_sent = max([len(data['Abstract']) for data in datas])
        max_len = max([min(len(sentence), self.max_len) for data in datas for sentence in data['Abstract']])
        batch_abstract = []
        batch_label = []
        sent_len = []
        for data in datas:
            # padding abstract to make them in same length
            pad_abstract = []
            for sentence in data['Abstract']:
                if len(sentence) > max_len:
                    pad_abstract.append(sentence[:max_len])
                else:
                    pad_abstract.append(sentence+[self.pad_idx]*(max_len-len(sentence)))
            sent_len.append(len(pad_abstract))
            pad_abstract.extend([[self.pad_idx]*max_len]*(max_sent-len(pad_abstract)))
            batch_abstract.append(pad_abstract)

            # gather labels
            if 'Label' in data:
                pad_label = data['Label']
                pad_label.extend([[0]*6]*(max_sent-len(pad_label)))
                batch_label.append(pad_label)
        
        return torch.LongTensor(batch_abstract), torch.FloatTensor(batch_label), sent_len


class simpleNet(nn.Module):
  def __init__(self, vocabulary_size):
        super(simpleNet, self).__init__()
        
        self.embedding_size = 50
        self.hidden_dim = 256
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_size)
        self.sent_rnn = nn.GRU(self.embedding_size,
                                self.hidden_dim,
                                bidirectional=True,
                                batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, 6)

  def forward(self, x):
        
        # x: (batch,sent,word)
        x = self.embedding(x)
        # x: (batch,sent,word,feature)
        b,s,w,e = x.shape
        x = x.view(b,s*w,e)
        # x: (batch,sent*word,feature)
        x, __ = self.sent_rnn(x)
        # x: (batch,sent*word,hidden_state*2)
        x = x.view(b,s,w,-1)
        # x: (batch,sent,word,hidden_state*2)
        x = torch.max(x,dim=2)[0]
        # x: (batch,sent,hidden_state*2)
        x = torch.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        # x: (batch,sent,6)
    
        return x







class F1():
  def __init__(self):
        self.threshold = 0.5
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

  def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

  def update(self, predicts, groundTruth):
        predicts = (predicts > self.threshold).float()
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth * predicts).data.item()

  def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20) #prevent divided by zero
        print(recall)
        print(precision)
        print(2 * (recall * precision) / (recall + precision + 1e-20))
        print("-----")
        return 2 * (recall * precision) / (recall + precision + 1e-20)

  def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

def _run_epoch(epoch, training):
    model.train(training)
    if training:
        description = 'Train'
        dataset = trainData
        shuffle = True
    else:
        description = 'Valid'
        dataset = validData
        shuffle = False
    dataloader = DataLoader(dataset=dataset,
                            batch_size=32,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn,
                            num_workers=0)

    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
    loss = 0
    f1_score = F1()
 
    for i, (x, y, sent_len) in trange:
        opt.zero_grad()

        abstract = x.to(device)
        labels = y.to(device)
        o_labels = model(abstract)
        batch_loss = criteria(o_labels, labels)

        if training:
            batch_loss.backward()
            opt.step()

        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), y)

        trange.set_postfix(
            loss=loss / (i + 1), f1=f1_score.print_score())
        
    if training:
        history['train'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
    else:
        history['valid'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})

def save(epoch):
    if not os.path.exists('model'):
        os.makedirs('model')
            
    torch.save(model.state_dict(), 'model.pkl.'+str(epoch))
    with open('model/history.json', 'w') as f:
        json.dump(history, f, indent=4)

def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
        sample = pd.read_csv(sampleFile) 
        submit = {}
        submit['order_id'] = list(sample.order_id.values)
        redundant = len(sample) - prediction.shape[0]
        if public:
            submit['BACKGROUND'] = list(prediction[:,0]) + [0]*redundant
            submit['OBJECTIVES'] = list(prediction[:,1]) + [0]*redundant
            submit['METHODS'] = list(prediction[:,2]) + [0]*redundant
            submit['RESULTS'] = list(prediction[:,3]) + [0]*redundant
            submit['CONCLUSIONS'] = list(prediction[:,4]) + [0]*redundant
            submit['OTHERS'] = list(prediction[:,5]) + [0]*redundant
        else:
            submit['BACKGROUND'] = [0]*redundant + list(prediction[:,0])
            submit['OBJECTIVES'] = [0]*redundant + list(prediction[:,1])
            submit['METHODS'] = [0]*redundant + list(prediction[:,2])
            submit['RESULTS'] = [0]*redundant + list(prediction[:,3])
            submit['CONCLUSIONS'] = [0]*redundant + list(prediction[:,4])
            submit['OTHERS'] = [0]*redundant + list(prediction[:,5])
        df = pd.DataFrame.from_dict(submit) 
        df.to_csv(filename,index=False)





def get_answer_tokens(filename):
    df = pd.read_csv(filename,dtype=str)
    tokens = []
    for i in df.iterrows():
        sents  = i[1]['Task 1'].split(" ")    
        tokens.append(sents)
    return tokens
    
def get_article_tokens(filename):
    df = pd.read_csv(filename,dtype=str)
    sentences= []
    for i in df.iterrows():
        sents = i[1]['Abstract'].split('$$$')
        sentences.append(sents)
    return sentences

def struct_get_token(tokens):
    tmp_tokens= []
    for i in range(30):

        
        tmp=[]
        for j in range(len(tokens)):
            
        
            if ( i == len(tokens[j]) ):
                flag = 1
                tmp.append(tokens[j])
                
                           
        tmp_tokens.append(tmp)
    return tmp_tokens

def struct_classify(tmp_tokens):
    sum_struct=[]
    
    for i in range(26):
        tmp_sum=[]
        
        for k in range(i):
            tmp = [0]*6
            

            for m in range(len(tmp_tokens[i])):

                if(tmp_tokens[i][m][k].find("BACKGROUND") != -1 ):
                    tmp[0]+=1
                if(tmp_tokens[i][m][k].find("OBJECTIVES") != -1):
                    tmp[1]+=1
                if(tmp_tokens[i][m][k].find("METHODS") != -1):
                    tmp[2]+=1
                if(tmp_tokens[i][m][k].find("RESULTS") != -1 ):
                    tmp[3]+=1
                if(tmp_tokens[i][m][k].find("CONCLUSIONS") != -1):
                    tmp[4]+=1
                if(tmp_tokens[i][m][k].find("OTHERS") != -1):
                    tmp[5]+=1
            tmp_sum.append(tmp)
        sum_struct.append(tmp_sum)
    return sum_struct

def struct_probability(sum_struct):
    for i in range(len(sum_struct)):
        for j in range(len(sum_struct[i])):
            a = sum_struct[i][j][0]
            b = sum_struct[i][j][1]
            c = sum_struct[i][j][2]
            d = sum_struct[i][j][3]
            e = sum_struct[i][j][4]
            f = sum_struct[i][j][5]
            total = a+b+c+d+e+f
            if(total==0):
                total = 1
            sum_struct[i][j]=[a/total,b/total,c/total,d/total,e/total,f/total]
    return sum_struct
def method_struct_classify(prob):
    tmp_answer=[]

    for i in range(len(prob)):
        tmp=[]
        for j in range(len(prob[i])):
            idex = prob[i][j].index(max(prob[i][j]))


            if(prob[i][j][idex]== 0):
                idex=(random.randint(0,5))
            if(prob[i][j][idex]<0.4):
                tmp.append("NONE")
                continue

            if(idex == 0):
                tmp.append(0)
            elif(idex == 1):
                tmp.append(1)
            elif(idex == 2):
                tmp.append(2)
            elif(idex == 3):
                tmp.append(3)
            elif(idex == 4):
                tmp.append(4)
            elif(idex == 5):
                tmp.append(5)
            
        tmp_answer.append(tmp)
    return tmp_answer      


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer_1(x)
        return x
class Dataset(Dataset):
  def __init__(self, data):
        self.data = data
  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        return self.data[index]
      
  def collate_fn(self, datas):
       
        return torch.FloatTensor(datas)

def get_one_hot(data_path):
    dataset = pd.read_csv(data_path, dtype=str)
    formatData = []
    for (idx,data) in dataset.iterrows():
        """
        processed: {
        'Abstract': [[4,5,6],[3,4,2],...]
        'Label': [[0,0,0,1,1,0],[1,0,0,0,1,0],...]
        }
        """
        processed=[]
        if ('Task 1' in data ):
            processed = [label_to_onehot(label) for label in data['Task 1'].split(' ')]
        formatData.append(processed)
    
    return formatData
                
def _net_epoch(epoch, training):
    net.train(training)
    
    dataloader = DataLoader(dataset=combine,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=combine.collate_fn,
                            num_workers=0)

    trange = tqdm(enumerate(dataloader), total=len(dataloader))
    loss = 0
    f1_score = F1()
    
    for i, (x) in trange:
        opt.zero_grad()
        abstract = x.to()
        labels = torch.FloatTensor(combine_labels[i]).view(1,1,-1)
        o_labels = net(abstract)
        batch_loss = criteria(o_labels, labels)
        print(trange)
        if training:
            batch_loss.backward()
            opt.step()

        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), labels.cpu())
        
        trange.set_postfix(
            loss=loss / (i + 1), f1=f1_score.print_score())
    
    if training:
        history['train'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
    else:
        history['valid'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
def net_save(epoch):
    if not os.path.exists('net'):
        os.makedirs('net')
            
    torch.save(net.state_dict(), 'net.pkl.'+str(epoch))
    with open('net/history.json', 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == '__main__':

    
    
    tokens = get_answer_tokens('abstract_and_task.csv')
    sentences = get_article_tokens('abstract_and_task.csv')
    tmp_tokens=struct_get_token(tokens)
    sum_struct = struct_classify(tmp_tokens)
    prob_struct = struct_probability(sum_struct) ##各個機率
    answer_struct = method_struct_classify(prob_struct) ##各個文章長度

    
    test_sentences =get_article_tokens('validset.csv')
    prob_every_article = []

    for i in range(len(test_sentences)):
        tmp=[]
        prob_every_article.append(prob_struct[len(test_sentences[i])])

    
    prob_every_article=np.array(prob_every_article) ##轉成np
    for i in range(len(prob_every_article)):
        prob_every_article[i]=np.array(prob_every_article[i])

    





    tmp_combine_labels=get_one_hot('validset.csv')
    combine_labels=[]
    for i in range(len(tmp_combine_labels)):
        for j in range(len(tmp_combine_labels[i])):
            combine_labels.append(tmp_combine_labels[i][j])

    
    


    words = set()
    words |= collect_words('trainset.csv')
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    word_dict = {'<pad>':PAD_TOKEN,'<unk>':UNK_TOKEN}
    for word in words:
        word_dict[word]=len(word_dict)

    with open('dicitonary.pkl','wb') as f:
        pickle.dump(word_dict, f)  
    
    with open('dicitonary.pkl','rb') as f:
        word_dict = pickle.load(f)
        


    print('[INFO] Start processing trainset...')
    train = get_dataset(r'trainset.csv', word_dict, n_workers=0)
    print(train[1]['Label'])
    print('[INFO] Start processing validset...')
    valid = get_dataset(r'validset.csv', word_dict, n_workers=0)


    confu_m =[]
    for i in range(len(valid)):
        confu_m.append(valid[i]['Label'])

    file = open('answer.txt','w')
    file.write(str(confu_m))
    file.close()

    print('[INFO] Start processing testset...')
    test = get_dataset(r'task1_public_testset.csv', word_dict, n_workers=0)


    

    trainData = AbstractDataset(train, PAD_TOKEN, max_len = 64)
    validData = AbstractDataset(valid, PAD_TOKEN, max_len = 64)
    testData = AbstractDataset(test, PAD_TOKEN, max_len = 64)

    
    
    
    device='cpu'
    
    model = simpleNet(len(word_dict))
    
    opt = torch.optim.Adam(model.parameters())
    criteria = torch.nn.BCELoss()


    """
    model.to(device)
    max_epoch = 10
    history = {'train':[],'valid':[]}
    
    
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, True)
        _run_epoch(epoch, False)
        save(epoch)
    with open('model/history.json', 'r') as f:
        history = json.loads(f.read())
    
    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7,5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.show()

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))
    """

    

    model.load_state_dict(torch.load('model.pkl.6',map_location=torch.device('cpu')))
    model.train(False)
    
    dataloader = DataLoader(dataset=validData, ##改
                                batch_size=1,
                                shuffle=False,
                                collate_fn=validData.collate_fn,
                                num_workers=0)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    
    prediction = []
    tmp_word=[]
    
    for i, (x, y, sent_len) in trange:
        
        o_labels = model(x.to(device))
        
        test = o_labels.cpu().detach().numpy().tolist()
        tmp_word.append(test)
        test=[]
        
        o_labels = o_labels>0.5
        
        for idx, o_label in enumerate(o_labels):
            prediction.append(o_label[:sent_len[idx]].to('cpu'))
            
    
    
    prediction = torch.cat(prediction).detach().numpy().astype(int)
    """
    conf_pr = []
    for i in range(len(prediction)):
        conf_pr.append(prediction[i])

    file = open('pred.txt','w')
    file.write(conf_pr)
    file.close()
    
    SubmitGenerator(prediction,'task1_sample_submission.csv',True, 'task1_search.csv')
    """
    
    
    
    new_word=list()
    for i in range(len(tmp_word)):
        for k in range(len(tmp_word[i])):
            new_word.append(tmp_word[i][k])

    
    

    for i in range(len(new_word)):
        new_word[i]=np.array(new_word[i])
        
    test_p=list()
    tmp=[]
    for i in range(len(new_word)):
        new_word[i]=prob_every_article[i]*1
        new_word[i]=new_word[i]>0.5
        new_word[i]=new_word[i].astype(int)
        new_word[i]=new_word[i].tolist()
        for j in range(len(new_word[i])):
            tmp.append(new_word[i][j])
    new_word=tmp
    

    combine_labels=list(chain.from_iterable(combine_labels))
    new_word=list(chain.from_iterable(new_word))
    test_f1=f1_score(combine_labels, new_word, average='macro')  
    reca_f1=recall_score(combine_labels, new_word, average='macro')
    prec_f1=precision_score(combine_labels, new_word, average='macro')
    print(test_f1)
    print(reca_f1)
    print(prec_f1)
    
    ##new_word是計算完的
    



    
            
    
    
    



    
    
    




    

    """
    net = Net(12,6)
    combine = Dataset(data_combine) ##轉成abstract
    opt = torch.optim.Adam(net.parameters())
    criteria = nn.MSELoss()
    net.to()
    max_epoch = 20
    history = {'train':[],'valid':[]}
    
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _net_epoch(epoch, True)
        
        net_save(epoch)
    with open('net/history.json', 'r') as f:
        history = json.loads(f.read())
    
    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7,5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.show()

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['train'])]))
    """
    


   
