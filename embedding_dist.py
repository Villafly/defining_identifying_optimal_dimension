'''
Reference implementation of node2vec. 

Author: Aditya Grover, modified by Weiwei Gu

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
import cPickle as pkl
#from define_identify import *

def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    #G = nx.read_edgelist(file_name, nodetype=int)
    if not args.directed:
        G = G.to_undirected()
    return G

def cal_cosine_matrices(G,walks,args):
    '''
    Compute the cosine distance between every node pair over different embedding dimensions.
    '''
    norm_loss = []
    walks = [map(str, walk) for walk in walks]
    embedding_dims = range(args.start_dim,args.end_dim,args.step)
    node_num = len(G.nodes())
#    #temp test
#    adj_list = np.asarray(nx.to_numpy_matrix(G))
#      
#    for i in range(len(G.nodes())):
#          for j in range(i,len(G.nodes())):
#            #print adj_list[i,j],adj_list[j,i],
#            if adj_list[i,j] != adj_list[j,i] or adj_list[i,i] == 1:              
#              print 'error',
#    # temp done
    if node_num < 500:
      embedding_dims.insert(0,node_num)
    else:
      embedding_dims.insert(0,500)  
    #cosine_matrices = np.zeros((len(embedding_dims),node_num,node_num)) 
    for _index, dim in enumerate(embedding_dims):
      model = Word2Vec(walks, size=dim,window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)    
      emb_matrix = np.zeros((node_num,dim))      
      for _cnt,node in enumerate(G.nodes()):
        emb_matrix[_cnt,:] = model[str(node)]
      cosine_matrix = cdist(emb_matrix,emb_matrix,'cosine')
      if _index == 0:
        benchmark_matrix = cosine_matrix
        benchmark_array = np.array(upper_tri_masking(benchmark_matrix))
        #np.savez_compressed('./pic/conect_data/npz/{}'.format(str.split(args.input,'/')[6]),benchmark_array)      
      else:
        dim_array = np.array(upper_tri_masking(cosine_matrix)) 
        loss = np.linalg.norm((dim_array-benchmark_array),ord=1)
        norm_loss.append(loss/len(dim_array))
    return norm_loss
    
def upper_tri_masking(A):
    '''
    Masking the upper triangular matrix. 
    '''
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]
  
def cal_embedding_distance(args):
    '''
    The overall random walk, graph embedding and cosine distance calculation process.
    '''
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.length)
    cosine_matrices = cal_cosine_matrices(nx_G,walks,args)
    return cosine_matrices


