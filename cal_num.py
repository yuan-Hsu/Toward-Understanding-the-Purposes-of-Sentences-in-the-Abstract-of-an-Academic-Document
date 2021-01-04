import pandas as pd
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
tokens=list()
df = pd.read_csv(r'abstract_and_task.csv',dtype=str)
for i in df.iterrows():
    sents  = i[1]['Task 1'].split(' ')
    tokens.append(sents)


bk=0
ob=0
me=0
re=0
con=0
oth=0

tmp=[]

structure=[]
for i in range(25):
    for j in range(i):
        tmp.append([0,0,0,0,0,0])
    structure.append(tmp)
    tmp=[]


for i in range(len(tokens)):
    for j in range(len(tokens[i])):
        if tokens[i][j].find("BACKGROUND")!=-1:
            structure[len(tokens[i])][j][0]+=1
        if tokens[i][j].find("OBJECTIVES")!=-1:
            structure[len(tokens[i])][j][1]+=1
        if tokens[i][j].find("METHODS")!=-1:
            structure[len(tokens[i])][j][2]+=1
        if tokens[i][j].find("RESULTS")!=-1:
            structure[len(tokens[i])][j][3]+=1
        if tokens[i][j].find("CONCLUSIONS")!=-1:
            structure[len(tokens[i])][j][4]+=1
        if tokens[i][j].find("OTHERS")!=-1:
            structure[len(tokens[i])][j][5]+=1
pred=list()


for i in range(25):
    for j in range(len(structure[i])):
        size=structure[i][j].index(max(structure[i][j]))
        if size==0:
            structure[i][j]=[1,0,0,0,0,0]
        if size==1:
            structure[i][j]=[0,1,0,0,0,0]
        if size==2:
            structure[i][j]=[0,0,1,0,0,0]
        if size==3:
            structure[i][j]=[0,0,0,1,0,0]
        if size==4:
            structure[i][j]=[0,0,0,0,1,0]
        if size==5:
            structure[i][j]=[0,0,0,0,0,1]

pred=list()
for i in range(len(tokens)):
    pred.append(structure[len(tokens[i])])


for i in range(len(structure)):
    print(i)
    print(structure[i])


"""
for i in range(len(tokens)):
    pred
"""

"""
for i  in range(len(tokens)):
    for j in range(len(tokens[i])):
        
        if tokens[i][j]=="BACKGROUND":
            bk=bk+1
        elif tokens[i][j]=="OBJECTIVES":
            ob=ob+1
        elif tokens[i][j]=="METHODS":
            me=me+1
        elif tokens[i][j]=="RESULTS":
            re=re+1
        elif tokens[i][j]=="CONCLUSIONS":
            con=con+1
        elif tokens[i][j]=="OTHERS":
            oth=oth+1
    
    tmp=[]

"""
new_pred=list() 
new_true=list()


for i  in range(len(tokens)):
    tmp=[]
    new_tmp=[]
    for j in range(len(tokens[i])):
        

        formal_answer=[0,0,0,0,0,0]

        if tokens[i][j].find("BACKGROUND")!=-1:
            formal_answer[0]=1
        if tokens[i][j].find("OBJECTIVES")!=-1:
            formal_answer[1]=1
        if tokens[i][j].find("METHODS")!=-1:
            formal_answer[2]=1
        if tokens[i][j].find("RESULTS")!=-1:
            formal_answer[3]=1
        if tokens[i][j].find("CONCLUSIONS")!=-1:
            formal_answer[4]=1
        if tokens[i][j].find("OTHERS")!=-1:
            formal_answer[5]=1
        
        new_true.append(formal_answer)
        new_pred.append(pred[i][j])





"""

new_true=np.array(new_true)
new_pred=np.array(new_pred)


print(new_true[1])
print(new_pred[1])
print(len(new_true))
print(len(new_pred))

titles=['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS','OTHERS']
print(sklearn.metrics.classification_report(new_true,new_pred,target_names=titles))
"""