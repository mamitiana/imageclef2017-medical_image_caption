
# coding: utf-8

# In[2]:

from PIL import Image
import os
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import transforms

from . import jsonLoader
import numpy as np
import pandas as pd
import config


# In[3]:
# In[4]:
# In[5]:



dtype= torch.cuda.FloatTensor
# In[6]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
class ImageClefDataset(Dataset):
    """Dataset wrapping images and target labels for ImageClef.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, img_ext, phase,target_transfo=None):
        ''' constructeur pour dataset '''
        if phase == "train": 
            jsdict=jsonLoader.TrainLoader()
        elif phase =="val":
            jsdict = jsonLoader.ValLoader()
        self.phase=phase
        dftemp=pd.DataFrame(jsdict.filtrer())
        print("image path"+img_path)
        assert dftemp['imageName'].apply(lambda x: os.path.isfile(os.path.join( img_path , x + img_ext ))).all(), "Some images referenced in the CSV file were not found"
        self.img_path = img_path
        self.img_ext = img_ext
        self.pred={}
        self.X_train = dftemp['imageName']
        self.caption = dftemp['caption']
        self.captionlist = dftemp['captions']
        self.target_transfo=target_transfo
       
    def getAnnotation(self,index):
        captiontxt = self.caption[index]
        captions=self.captionlist[index]
        return  captiontxt , captions , index
    
    def getImage(self,index):
        filepath=os.path.join(self.img_path, self.X_train[index] + self.img_ext )
        img = Image.open( filepath )
        img = img.convert('RGB')
        img = data_transforms[self.phase](img)
        return img
    
    
    def __getitem__(self, index):

        img = self.getImage(index)
        captiontxt , captions , index = self.getAnnotation(index)
        if self.target_transfo is not None:
            captions = self.target_transfo(captions)
        

        
        return img  , captions , index


    def __len__(self):
        ''' longeur  '''
        return len(self.X_train.index)
    
    

        
        
    def exportPrediction(self,path):
        temp =pd.DataFrame.from_dict(self.pred,orient='index')
        temp.reset_index( inplace=True)
        print(temp)
        #dfval['classpredstr']=dfval['classpred'].apply(lambda x:','.join(x))
        temp.to_csv(path,index=False,header=False,sep="\t")


# In[7]:

if __name__ == '__main__':

    imp="/home/mamitiana/imageclef/detection/train/image"
    clef= ImageClefDataset(imp,".jpg",'train')
    for i in range(len(clef)):
        im,lab=clef[4]
        if type(im) is not torch.FloatTensor:
            print(type(im))
    print("fin,all of them are tesor")


# In[ ]:



