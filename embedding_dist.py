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
    walks = [map(str, walk) for walk in walks]
    embedding_dims = range(args.start_dim,args.end_dim,args.step)
    node_num = len(G.nodes())
    embedding_dims.append(node_num)  
    cosine_matrices = np.zeros((len(embedding_dims),node_num,node_num)) 
    for _index, dim in enumerate(embedding_dims):
      model = Word2Vec(walks, size=dim,window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)    
      emb_matrix = np.zeros((node_num,dim))
      for _cnt,node in enumerate(G.nodes()):
        emb_matrix[_cnt,:] = model[str(node)]
      cosine_matrix = cdist(emb_matrix,emb_matrix,'cosine')
      cosine_matrices[_index,:,:] = cosine_matrix
    #print np.shape(cosine_matrices)
    return cosine_matrices
  
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


