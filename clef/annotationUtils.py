'''
Created on 25 août 2017

@author: mamitiana
'''
from random import randrange
import string
import torch

from clef.datasetSync import ImageClefDataset
import config


def word_tokenize(captions):
    ''' tokenize caption to word '''
    processed = []
    for j, s in enumerate(captions):
        txt = str(s).lower().translate(
            string.punctuation).strip().split()
        processed.append(txt)
    return processed


def build_vocab(dset, num_words=-1):
    # count up the number of words
    counts = {}

    for img_id in  range (len(dset) ):
        

        _ ,  captionlist , _ = dset[img_id]
        
        captions = word_tokenize([ann for ann in captionlist]) # tokenize the caption
        for txt in captions:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for (_, w) in cw[:num_words]]
 
    vocab = [config.Configuration.PAD_TOKEN] + vocab + [config.Configuration.UNK_TOKEN, config.Configuration.EOS_TOKEN]

    return vocab



def create_target(vocab, rnd_caption=True):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    unk = word2idx[config.Configuration.UNK_TOKEN]

    def get_caption(captions):
        captions = word_tokenize(captions)
        if len(captions) == 0:
            return torch.Tensor([unk])
        if rnd_caption:
            
            idx = randrange(len(captions))
        else:
            idx = 0
        caption = captions[idx]
        targets = []
        for w in caption:
            targets.append(word2idx.get(w, unk))
        return torch.Tensor(targets)
    return get_caption
    
def create_collate(vocab, max_length=50):
    padding = vocab.index(config.Configuration.PAD_TOKEN)
    eos = vocab.index(config.Configuration.EOS_TOKEN)

    def collate(img_cap):
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)

        imgs, caps ,_ = zip(*img_cap)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos

            cap_tensor[:end_cap, i].copy_(c[:end_cap])

        return (imgs, (cap_tensor, lengths))
    return collate



def get_iterator(data, vocab, batch_size=32, max_length=30, shuffle=True, num_workers=4, pin_memory=True):
   
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=create_collate(vocab, max_length),
        num_workers=num_workers, pin_memory=pin_memory)
    
    

if __name__ == '__main__':
    val=ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val')
    print(len(build_vocab(val)))