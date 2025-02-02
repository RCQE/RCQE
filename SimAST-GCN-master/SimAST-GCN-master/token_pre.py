# project = ['accumulo','ambari','cloudstack','commons-lang','flink']
project = ['ref']

import pandas as pd
from gensim.models.word2vec import Word2Vec
import multiprocessing
import os

def transform(tmp):
    tmp = tmp.replace("'",' ').replace(':',' ').replace('"',' ').replace('\n',' ').replace('?',' ').replace(';',' ').replace('(',' ').replace('.',' ').replace(')',' ').replace('{',' ').replace('}',' ').replace('=',' ').replace('[',' ').replace(']',' ').replace('<',' ').replace('>',' ').replace(',',' ').replace('+',' ').replace('-',' ').replace('/',' ').replace('*',' ').replace('!',' ')
    tmp = tmp.split()
    #tmp = ' '.join(tmp)
    return tmp

def tf(tmp):
    tmp = tmp.replace("'",' ').replace(':',' ').replace('"',' ').replace('\n','$').replace('?',' ').replace(';',' ').replace('(',' ').replace('.',' ').replace(')',' ').replace('{',' ').replace('}',' ').replace('=',' ').replace('[',' ').replace(']',' ').replace('<',' ').replace('>',' ').replace(',',' ').replace('+',' ').replace('-',' ').replace('/',' ').replace('*',' ').replace('!',' ')
    # tmp = tmp.split()
    # print(tmp)
    # tmp = ' '.join(tmp)
    # ever_alp = 0
    # is_sp = 0
    # tmp = list(tmp)
    # for i in range(len(tmp)):
    #     if (tmp[i]>='a' and tmp[i]<='z') or (tmp[i]>='A' and tmp[i]<='Z'):
    #         ever_alp = 1
    #         is_sp = 0
    #     if tmp[i]=='$' and ever_alp==0:tmp[i]=' '
    #     if ever_alp==1:
    #         if tmp[i]=='$' and is_sp==0:is_sp = 1
    #         elif tmp[i]=='$' and is_sp==1:tmp[i]=' '
    # if tmp[-1]=='$':tmp[-1]=' '
    # tmp = ''.join(tmp) 
    # tmp = tmp.split()
    # tmp = ' '.join(tmp)
    tmp = tmp.replace(' $ ','$')
    # while not ( (tmp[-1]>='a' and tmp[-1]<='z') or (tmp[-1]>='A' and tmp[-1]<='Z') ):
    #     tmp = tmp[:-1]
    return tmp



for p in project:
    ratio = '3:1:1'
    print('procesing :',p)
    path = 'data/'+p+'/'+p+'.pkl'
    s = pd.read_pickle(path)
    s = pd.DataFrame(s)
    print(len(s))
    s.columns = ['cmt','label','old','new']
    corpus = s['old'].apply(transform)
    corpus += s['new'].apply(transform)
    w2v = Word2Vec(corpus, vector_size=128, sg=1, window=5 ,min_count = 3, workers=multiprocessing.cpu_count()) # max_final_vocab=3000 tmp ignore
    if not os.path.exists('data/'+p+'/token'):
        os.mkdir('data/'+p+'/token')
    w2v.save('data/'+p+'/token/node_w2v_128')

    word2vec = w2v.wv
    vocab = word2vec.key_to_index
    # print(word2vec.vectors.shape[0])
    # print(word2vec.vectors.shape[5])
    max_token = word2vec.vectors.shape[0]
    # max_token = 500
    def to_index(tmp):
        # print(tmp)
        tmp = tf(tmp)
        sen = tmp.split('$')
        idx = []
        flag = 0
        for _ in sen:
            lst = _.split()
            tk = []
            for __ in lst:
                tk.append(vocab[__] if __ in vocab else max_token)
            idx.append(tk)
            if len(tk)>100:flag = 1
        if len(idx)>50:flag = 1
        if flag==1:return []
        for x in idx:
            while len(x)!=100:x.append(max_token)
        tc = [max_token for i in range(100)]
        while len(idx)!=50:idx.append(tc)
        return [n for x in idx for n in x]
    
    s['old'] = s['old'].apply(to_index)
    s['new'] = s['new'].apply(to_index)

    dellist = []
    for _,item in s.iterrows():
        if len(item['old']) == 0 or len(item['new']) == 0 :dellist.append(_)

    s.drop(dellist,inplace = True)

    data_num = len(s)
    ratios = [int(r) for r in ratio.split(':')]
    train_split = int(ratios[0]/sum(ratios)*data_num)
    val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

    s = s.sample(frac=1, random_state=666)
    train = s.iloc[:train_split]
    dev = s.iloc[train_split:val_split]
    test = s.iloc[val_split:]

    trainp = 'data/'+p+'/token/'+p+'_token_train.pkl'
    testp = 'data/'+p+'/token/'+p+'_token_test.pkl'
    devp = 'data/'+p+'/token/'+p+'_token_dev.pkl'
    pp = 'data/'+p+'/token/'+p+'_token.pkl'
    s.to_pickle(pp)
    train.to_pickle(trainp)
    test.to_pickle(testp)
    dev.to_pickle(devp)
