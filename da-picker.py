from argparse import ArgumentParser
from termcolor import colored

#internal imports
from da_utils import ModelUtils
from da_algos import DaTechniques

import shutil
import torch 



def driver():
  parser = ArgumentParser()
  parser.add_argument("--INPUT_FILE_PATH", type=str, help="path to PyTorch model")
  parser.add_argument("--OUTPUT_FILE_PATH", type=str, help="path to store DA Output Dataset")
  parser.add_argument("--subset_size", default = 0.1, type=float, help="subset size for DA")
  parser.add_argument("--metric", default = False, type=bool, help="enable count metric")
  parser.add_argument("--DATA_PATH", type=str, default=None, help="Path to dataset in ImageFolder format")
  parser.add_argument("--UMAP", default = False, type=bool, help="enable UMAP")
  parser.add_argument("--img_size", default = None, type=int, help="Size of Images (Required when passing model)")
  parser.add_argument("--embedding_size", default = None, type=int, help="Size of model's output embedding (Required when passing model)")
  parser.add_argument("--technique", type=str, default="DA_STD", help="Diversity Algorithm Technique (DA_STD/DA_FAST)")
  parser.add_argument("--sample_size", type=float, default=None, help="Needed if DA_FAST is called. Number of elements to be randomly sampled from the subset")

  args = parser.parse_args()
  # INPUT_FILE_PATH = args.INPUT_FILE_PATH
  # OUTPUT_FILE_PATH = args.OUTPUT_FILE_PATH
  # subset_size = args.subset_size
  # metric = args.metric
  # UMAP = args.UMAP
  # DATA_PATH = args.DATA_PATH
  # img_size = args.img_size
  # embedding_size = args.embedding_size
  # technique = args.technique
  # sample_size = args.sample_size

  utils = ModelUtils(args.INPUT_FILE_PATH, args.DATA_PATH, args.img_size, args.embedding_size)
  if '.pt' in args.INPUT_FILE_PATH:
    filenames, feature_list = utils.get_embeddings()

  
  print(colored("Number of Files and Features",'blue'),len(filenames), colored("Embedding Size",'blue'), feature_list[0].shape)
  da = DaTechniques(args.subset_size, filenames, feature_list, args.sample_size)
  
  
  if args.technique=='DA_STD':
    print(colored("Running DA Standard..",'blue'))
    da_files, da_embeddings, _ = da.min_max_diverse_embeddings(i = da.farthest_point())
  elif args.technique=='DA_FAST':
      print(colored("Running DA Fast..",'blue'))
      da_files, da_embeddings, _ = da.min_max_diverse_embeddings_fast(i = da.farthest_point())
  utils.filenames = da_files
  utils.embeddings = da_embeddings
  if args.OUTPUT_FILE_PATH != None:
      utils.prepare_dataset(args.OUTPUT_FILE_PATH)

  if args.metric:
      utils.class_distrib()
  if args.UMAP:
      utils.plot_umap(args.OUTPUT_FILE_PATH)

if __name__ == "__main__":
  driver()
