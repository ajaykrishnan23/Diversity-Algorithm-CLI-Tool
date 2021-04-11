from argparse import ArgumentParser
from termcolor import colored
from numba import jit,njit
from tqdm import tqdm
import scipy.spatial.distance as dist
import numpy as np
import random

class DaTechniques:
  def __init__(self, subset_size, filenames, embeddings, sample_size=None):
    
    self.filenames = filenames
    self.embeddings = embeddings
    self.subset_size = subset_size
    self.sample_size = sample_size


  def farthest_point(self):
    
    centroid = sum(self.embeddings)/len(self.embeddings)
    distances = [dist.euclidean(i, centroid) for i in self.embeddings]
    return distances.index(max(distances))

  def min_max_diverse_embeddings(self):
    #calls Diversity_Algorithm
    if len(self.embeddings) != len(self.filenames) or len(self.embeddings) == 0 :
      return 'Data Inconsistent'
    else:
      return self.compute_diversity(self.filenames, self.embeddings, self.subset_size, i=self.farthest_point())


  @staticmethod
  @jit(nopython=True)
  def compute_diversity(filenames, embeddings, subset_size, i = None) :

    n = int(subset_size * len(embeddings))
    # print("Len of Filenames and Feature List for sanity check:",len(filenames),len(feature_list))
    print("Running DA Standard..")
    print("Subset_Size:",n)
    filename_copy = filenames
    set_input = embeddings
    set_output = []
    filename_output = []
    idx = 0
    if i is None: 
        idx = random.randint(0, len(set_input) -1)
    else:
        idx = i
    # set_output = np.append(set_output,set_input[idx])
    set_output.append(set_input[idx])
    filename_output.append(filename_copy[idx])
    min_distances = np.full(len(set_input),np.inf)
    # maximizes the minimum distance
    count = 10
    for _ in range(n - 1):
        if _>count:
          print(count,"Samples Done")
          count+= 500
        for i in range(len(set_input)) :
            dist = np.linalg.norm(set_input[i] - set_output[-1])
            if min_distances[i] > dist :
                min_distances[i] = dist
        inds = np.argmax(min_distances)
        # inds = min_distances.index(max(min_distances))
        # print(inds)
        set_output.append(set_input[inds])
        filename_output.append(filename_copy[inds])
    return filename_output, set_output, min_distances

  def min_max_diverse_embeddings_fast(self):
    #calls Diversity_Algorithm
    if len(self.embeddings) != len(self.filenames) or len(self.embeddings) == 0 :
      return 'Data Inconsistent'
    else:
      return self.compute_diversity_fast(self.filenames, self.embeddings, self.subset_size, self.sample_size, i=self.farthest_point())


  @staticmethod
  @jit(nopython=True)
  def compute_diversity_fast(filenames, embeddings, subset_size, sample_size, i = None) :
    def random_sample(embeddings_len, sample):
      return np.random.choice(embeddings_len,sample,replace=False)
      
    n = int(subset_size * len(embeddings))
    sample_size = int(sample_size * len(filenames))
    print("Running DA Fast..")
    print("Subset Size:",n)
    print("Sample Size:",sample_size)
    filename_copy = filenames
    set_input = embeddings
    set_output = []
    filename_output = []
    idx = 0
    if i is None: 
        idx = random.randint(0, len(set_input) -1)
    else:
        idx = i
    set_output.append(set_input[idx])
    filename_output.append(filename_copy[idx])
    min_distances = np.full(len(set_input),np.inf)
    # maximizes the minimum distance
    count = 10
    for _ in (range(n - 1)):
      #tqdm alternative
      if _>count:
        print(count,"Samples Done")
        count+= 500
      sampled_inds = random_sample(len(embeddings),sample_size)
      for idx in sampled_inds :  
          dist = np.linalg.norm(set_input[idx] - set_output[-1])
          if min_distances[idx] > dist :
              min_distances[idx] = dist
      inds = np.argmax(min_distances)
      set_output.append(set_input[inds])
      filename_output.append(filename_copy[inds])
      dist = np.linalg.norm(set_input[inds] - set_output[-1])
      if min_distances[inds]>dist:
        min_distances[inds] = dist 

    return filename_output, set_output, min_distances
