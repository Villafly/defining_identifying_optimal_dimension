import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import math
import argparse

from embedding_dist import cal_embedding_distance

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--input', nargs='?', default='./graph/cora',
                        help='Input graph path')
    
    parser.add_argument('--output', nargs='?', default='./emb',
                        help='Embeddings path')
                                           
    parser.add_argument('--start_dim', type=int, default= 2,
                        help='the start embedding dimension. Default is 2.')
                                           
    parser.add_argument('--end_dim', type=int, default= 180,
                        help='the end embedding dimension. Default is 180.')
                                           
    parser.add_argument('--step', type=int, default= 2,
                        help='the step dimension from start_dim to end_dim. Default is 2.')                                        
     
    parser.add_argument('--length', type=int, default= 10,
                        help='Length of walk per source. Default is 80.')
    
    parser.add_argument('--num-walks', type=int, default=50,
                        help='Number of walks per source. Default is 10.')
    
    parser.add_argument('--window_size', type=int, default=5,
                      	help='Context size for optimization. Default is 10.')
    
    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')
    
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
                                           
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
                                           
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()
 
def upper_tri_masking(A):
  	'''
  	Masking the upper triangular matrix. 
  	'''
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]
    
def fitting_func(dims,s,L):  
  return s/dims + L
 
def define_loss(cosine_matrices):
	'''
	Compute the normalized embedding loss of different embeddings. 
	'''
  benchmark_matrix = cosine_matrices[-1,:,:]
  benchmark_array = np.array(upper_tri_masking(benchmark_matrix))
  norm_loss = []
  for dim_matrix in cosine_matrices[:-1,:,:]: 
    dim_array = np.array(upper_tri_masking(dim_matrix)) 
    loss = np.linalg.norm((dim_array-benchmark_array),ord=1)
    norm_loss.append(loss/len(dim_array))
  return norm_loss  
  
def identify_optimal_dim(loss,args):
	'''
	Identify the optimal dimension range and compute the curve fitting parameter for graph.
	'''
  dims = range(args.start_dim,args.end_dim,args.step)
  paras,cov = optimize.curve_fit(fitting_func, dims,loss)
  fit_values = (fitting_func(np.array(dims),paras[0],paras[1]))
  SSE = ((np.array(loss)-np.array(fit_values))**2).mean()
  print 'the optimal dimension ranges from {} to {}'.format(int(round(paras[0]/0.03)),int(round(paras[0]/0.02)))
  print 'the Sum of Squares Error of curve fitting is {}'.format(SSE)
  print 's value is {}'.format(paras[0])
  plt.scatter(dims,loss) 
  plt.plot(dims,fit_values,c='r')
  plt.savefig('./pic/{}.png'.format(str.split(args.input,'/')[2]),format='png')
  plt.show()
 

if __name__ == "__main__":
    args = parse_args()
    cosine_matrices = cal_embedding_distance(args)
    norm_loss = define_loss(cosine_matrices)
    identify_optimal_dim(norm_loss,args)
