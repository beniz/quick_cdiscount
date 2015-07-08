# -*- coding: utf-8 -*-

import os,sys,csv,itertools
from gensim.models import word2vec

def proc_txt(txt):
    ntxt = txt.translate(None,'()[]{}')
    ntxt = ntxt.replace(' x ',' ').replace(' - ',' ').replace('_ ',' ').replace('...','').replace('/',' ').replace(',',' ').replace('.',' ').replace('…',' ').replace('\'','').replace('"','').replace('!','').replace('?','').replace('°','').replace(' -','').replace('+',' ').replace(':',' ')
    return ntxt

def w2v(txt,model,embed_size):
    ws = txt.split()
    lv = []
    for w in ws:
        try:
            v = model[w]
        except:
            v = [0.0]*embed_size
        lv.append(v)
    vf = [sum(sublist) for sublist in itertools.izip(*lv)]
    return vf

def w2frow(nrow,txt,model,embed_size):
    ntxt = proc_txt(txt)
    vf = w2v(ntxt,model,embed_size)
    for v in vf:
        nrow.append(v)

csvfile = 'training_shuf.csv'  # put your training file here
outcsvfile = 'training_w2v_shuf.csv' # put your training output file here
w2vmodel = 'cdis.bin' # word2vec model

fl = open(csvfile,'r')
flo = open(outcsvfile,'w')
lreader = csv.reader(fl,delimiter=';')
lwriter = csv.writer(flo,delimiter=';')
allcat3 = {}

embed_size = 200
model = word2vec.Word2Vec.load_word2vec_format(w2vmodel,binary=True) # Beware: this requires patching gensim loader, contact me if you're having issues

i = 0
for row in lreader:
    if i == 0:
        i = i + 1
        nrow = []
        nrow.append(row[0])
        nrow.append(row[3])
        for i in range(embed_size):
            nrow.append('df'+str(i)) # description
        nrow.append(row[8])
        lwriter.writerow(nrow)
        continue
    nrow = []
    nrow.append(row[0])
    cat3 = row[3]
    if not cat3 in allcat3:
        allcat3[cat3] = len(allcat3)
    desc = row[4].lower()
    libel = row[5].lower()
    marque = row[6].lower()
    if marque == 'aucune':
        marque = ''
    if row[8] == -1:
        row[8] = 0
    nrow.append(allcat3[cat3])
    w2frow(nrow,desc+' '+libel+' '+marque,model,embed_size)
    nrow.append(row[8])
    
    lwriter.writerow(nrow)

nclasses = len(allcat3)

print allcat3
print 'number of classes=',nclasses
print allcat3
