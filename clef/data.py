'''
Created on 25 août 2017

@author: mamitiana
'''
from clef.annotationUtils import build_vocab
from clef.datasetSync import ImageClefDataset
import config


class Data(object):
    '''
    fusion de datasetSync pour train and val
    '''


    def __init__(self):
        '''
        Constructor
        '''
        #self.train= ImageClefDataset(config.Configuration.trainImages,config.Configuration.trainext,'train') ,
        temp=ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val')
                
    
       
        
        self.vocab = build_vocab(temp)
        
        
d= Data()
print(d.vocab[:200])

        
        