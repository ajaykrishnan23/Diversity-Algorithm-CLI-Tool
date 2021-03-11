from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from torchvision import transforms
from termcolor import colored
from matplotlib import colors
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np 
import shutil
import pickle
import torch
import umap
import os



class ModelUtils:
  def __init__(self, MODEL_PATH=None, DATA_PATH=None, img_size=None, embedding_size=None, embeddings=None,filenames=None):
    
    self.MODEL_PATH = MODEL_PATH
    self.DATA_PATH = DATA_PATH
    self.img_size = img_size
    self.embedding_size = embedding_size
    self.embeddings = embeddings 
    self.filenames = filenames


  def get_embeddings(self):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()
    t = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.Lambda(to_tensor)
                            ])
    dataset = ImageFolder(self.DATA_PATH, transform = t)
    model = torch.load(self.MODEL_PATH)
    model.eval()
    model.cuda()
    with torch.no_grad():
        data_matrix = torch.empty(size = (0, self.embedding_size)).cuda()
        bs = 32
        if len(dataset) < bs:
          bs = 1
        loader = DataLoader(dataset, batch_size = bs, shuffle = False)
        for batch in loader:
            x = batch[0].cuda()
            embeddings = model(x)[0]
            data_matrix = torch.vstack((data_matrix, embeddings))
    paths = [dataset.imgs[i][0] for i in range(len(dataset.imgs))]
    self.embeddings = data_matrix.cpu().detach().numpy()
    self.filenames = paths #should i save this as class variables or return them alone?
    print(colored('Embeddings Extracted...','blue'))
    return paths, data_matrix.cpu().detach().numpy()
  
  def class_distrib(self):
    classes = {}
    for x in self.filenames:
      if x.split('/')[-2] in classes:
        classes[x.split('/')[-2]] += 1
      else:
        classes[x.split('/')[-2]] = 1

    classes_dict = {}
    for _ in classes.keys():
      print(colored(_,'green') , '= ', colored(classes[_],'green'))
      classes_dict[_] = classes[_]
    x_distrib = classes_dict

    x_std = np.std(list(classes.values()))
    print(colored('\nStandard Deviation between classes ','green'), colored(x_std,'green'))

    x_diff = max(list(classes.values()))- min(list(classes.values()))
    print(colored('\nDifference between max and minimum values ','green'), colored(x_diff,'green'))
    return x_distrib, x_std, x_diff 

  def prepare_dataset(self,dir): 
    try: 
      os.makedirs(dir)
    except:
      print("Directory already exists. Deletinga and creating new directory.")
      shutil.rmtree(dir)
      os.makedirs(dir)
      # os.makedirs(os.path.join(dir, 'train'))

    for x in self.filenames:
      # if x.split("/")[-2] not in os.listdir(os.path.join(dir,"train")):
      if x.split("/")[-2] not in os.listdir(dir):
        # os.mkdir(os.path.join(os.path.join(dir,"train"),x.split("/")[-2]))
        os.mkdir(os.path.join(dir,x.split("/")[-2]))
        shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
      else:
        shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
    print(colored('Subset moved to directory. Dataset abiding formats created successfully','blue'))

  def plot_umap(self, output_path, n_neighbors=20):
    class_id = []
    for _ in self.filenames:
      class_id.append(_.split("/")[-2])
    num_points = dict((x,class_id.count(x)) for x in set(class_id))
    txt = ''
    for i in num_points.keys():
      txt += i + ':' + str(num_points[i]) + " "
    le = LabelEncoder()
    class_labels = le.fit_transform(class_id)

    fit = umap.UMAP(
          n_neighbors=n_neighbors,
          n_components=2,
          metric='euclidean')

    u = fit.fit_transform(self.embeddings)
    color_map = plt.cm.get_cmap('tab20b_r')
    scatter_plot= plt.scatter(u[:,0], u[:, 1], c=class_labels, cmap = color_map)
    plt.title('UMAP embedding');
    plt.colorbar(scatter_plot)
    
    fname = output_path + '_UMAP_DA_Embeddings' + '.png'
    print(colored("UMAP Saved at:",'blue'),fname)
    plt.savefig(fname)
    plt.show();

  