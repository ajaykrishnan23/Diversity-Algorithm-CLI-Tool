from argparse import ArgumentParser
from termcolor import colored
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

  def min_max_diverse_embeddings(self, i = None) :

    if len(self.embeddings) != len(self.filenames) or len(self.embeddings) == 0 :
        return 'Data Inconsistent'
    n = int(self.subset_size * len(self.embeddings))
    # print("Len of Filenames and Feature List for sanity check:",len(filenames),len(feature_list))
    print("Running DA Standard..")
    print("Subset_Size:",n)
    filename_copy = self.filenames.copy()
    set_input = self.embeddings.copy()
    set_output = []
    filename_output = []
    idx = 0
    if i is None: 
        idx = random.randint(0, len(set_input) -1)
    else:
        idx = i
    set_output.append(set_input[idx])
    filename_output.append(filename_copy[idx])
    min_distances = [1000] * len(set_input)
    # maximizes the minimum distance
    for _ in tqdm(range(n - 1)):
        for i in range(len(set_input)) :
            # distances[i] = minimum of the distances between set_output and one of set_input
            dist = np.linalg.norm(set_input[i] - set_output[-1])
            if min_distances[i] > dist :
                min_distances[i] = dist
        inds = min_distances.index(max(min_distances))
        set_output.append(set_input[inds])
        filename_output.append(filename_copy[inds])

    return filename_output, set_output, min_distances

  def min_max_diverse_embeddings_fast(self, i = None) :

    if len(self.embeddings) != len(self.filenames) or len(self.embeddings) == 0 :
        return 'Data Inconsistent'
    n = int(self.subset_size * len(self.embeddings))
    sample_size = int(self.sample_size * len(self.filenames))
    print("Subset Size:",n)
    print("Sample Size:",sample_size)
    # print("Len of Filenames and Feature List for sanity check:",len(filenames),len(feature_list))
    filename_copy = self.filenames.copy()
    set_input = self.embeddings.copy()
    set_output = []
    filename_output = []
    idx = 0
    if i is None: 
        idx = random.randint(0, len(set_input) -1)
    else:
        idx = i
    set_output.append(set_input[idx])
    filename_output.append(filename_copy[idx])
    min_distances = [1000] * len(set_input)
    # maximizes the minimum distance
    count = 0
    for _ in tqdm(range(n - 1)):
      # min_distances[inds] = (set_input[inds] - set_output[-1]) #incase the index value of maximum index does not get selected
      sampled_inds = random.sample(range(0,len(self.embeddings)),sample_size)
      for idx in sampled_inds :  
          dist = np.linalg.norm(set_input[idx] - set_output[-1])
          if min_distances[idx] > dist :
              min_distances[idx] = dist
      inds = min_distances.index(max(min_distances))
      set_output.append(set_input[inds])
      filename_output.append(filename_copy[inds])
      dist = np.linalg.norm(set_input[inds] - set_output[-1])
      if min_distances[inds]>dist:
        min_distances[inds] = dist 

    return filename_output, set_output, min_distances
