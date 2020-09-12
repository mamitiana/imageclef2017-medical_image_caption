'''
Created on 25 août 2017

@author: mamitiana
'''
from datetime import datetime
import logging
import math
import os
import time
import torch
from torch.autograd.variable import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

from clef.annotationUtils import build_vocab
from clef.annotationUtils import create_target, get_iterator
from clef.datasetSync import ImageClefDataset
from clef.utils import setup_logging, AverageMeter, select_optimizer, \
    adjust_optimizer
import config
from model import CaptionModel
import torch.backends.cudnn as cudnn
import torch.nn as nn


def forward(model, data, epoch,training=True, optimizer=None):
        use_cuda = 'cuda' in type
        loss = nn.CrossEntropyLoss()
        perplexity = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if training:
            model.train()
        else:
            model.eval()

        end = time.time()
        for i, (imgs, (captions, lengths)) in enumerate(data):
            data_time.update(time.time() - end)
            if use_cuda:
                imgs = imgs.cuda()
                captions = captions.cuda(async=True)
            imgs = Variable(imgs, volatile=not training)
            captions = Variable(captions, volatile=not training)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]

            pred, _ = model(imgs, input_captions, lengths)
            err = loss(pred, target_captions)
            perplexity.update(math.exp(err.data[0]))

            if training:
                optimizer.zero_grad()
                err.backward()
                clip_grad_norm(model.rnn.parameters(), grad_clip)
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(
                                 epoch, i, len(data),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, perp=perplexity))
                
                
if __name__ == '__main__':
    cnn = models.resnet18(pretrained=True)
    save=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir='./results'
    save_path = os.path.join(results_dir, save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    setup_logging(os.path.join(save_path, 'log.txt'))
    
    checkpoint_file = os.path.join(save_path, 'checkpoint_epoch_%s.pth.tar')
    embedding_size=256 
    rnn_size=256
    num_layers=2 #nombre de couche rnn
    max_length=50 # nombre de mots max
    type="torch.cuda.FloatTensor"
    workers=8
    epochs=10
    finetune_epoch=3
    start_epoch=0
    batch_size=32
    eval_batch_size=128
    optimizer='SGD'
    grad_clip=5
    learning_rate=0.1
    lr_decay=0.8
    momentum=0.9
    weight_decay=1e-4
    share_weights=False
    print_freq=10
    
    # mot pour train
    temp= ImageClefDataset(config.Configuration.trainImages,config.Configuration.trainext,'train') 
    # mot pour val
    #temp=ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val')
    vocab = build_vocab(temp)
    print('# words: '+str(len(vocab)))
    
    del temp
    trainDset = ImageClefDataset(config.Configuration.trainImages,config.Configuration.trainext,'train',target_transfo=create_target(vocab))
    valDset=ImageClefDataset(config.Configuration.valImages,config.Configuration.valext,'val',target_transfo=create_target(vocab))
    
    trainDataLoader =get_iterator(trainDset,vocab,
                            batch_size=batch_size,
                              max_length=max_length,
                              shuffle=True,
                              num_workers=workers
                                  )
    print("trainloader ok")
    
    valDataLoader = get_iterator(valDset , vocab ,
                            batch_size=eval_batch_size,
                            max_length=max_length,
                            shuffle=False,
                            num_workers=workers)


    model = CaptionModel(cnn, vocab,
                         embedding_size=embedding_size,
                         rnn_size=rnn_size,
                         num_layers=num_layers,
                         share_embedding_weights=share_weights)
    
    
    
    
    if 'cuda' in type:
        cudnn.benchmark = True
        model.cuda()

    optimizer = select_optimizer(
        optimizer, params=model.parameters(), lr=learning_rate)
    regime = lambda e: {'lr': learning_rate * (lr_decay ** e),
                        'momentum': momentum,
                        'weight_decay': weight_decay}
    model.finetune_cnn(False)
    def forward(model, data, training=True, optimizer=None):
        use_cuda = 'cuda' in type
        loss = nn.CrossEntropyLoss()
        perplexity = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if training:
            model.train()
        else:
            model.eval()

        end = time.time()
        for i, (imgs, (captions, lengths)) in enumerate(data):
            data_time.update(time.time() - end)
            if use_cuda:
                imgs = imgs.cuda()
                captions = captions.cuda(async=True)
            imgs = Variable(imgs, volatile=not training)
            captions = Variable(captions, volatile=not training)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]

            pred, _ = model(imgs, input_captions, lengths)
            err = loss(pred, target_captions)
            perplexity.update(math.exp(err.data[0]))

            if training:
                optimizer.zero_grad()
                err.backward()
                clip_grad_norm(model.rnn.parameters(), grad_clip)
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(
                                 epoch, i, len(data),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time, perp=perplexity))

        return perplexity.avg

    for epoch in range(start_epoch, epochs):
        if epoch >= finetune_epoch:
            model.finetune_cnn(True)
        optimizer = adjust_optimizer(
            optimizer, epoch, regime)
        # Train
        train_perp = forward(
            model, trainDataLoader, training=True, optimizer=optimizer)
        # Evaluate
        val_perp = forward(model, valDataLoader, training=False)

        logging.info('\n Epoch: {0}\t'
                     'Training Perplexity {train_perp:.4f} \t'
                     'Validation Perplexity {val_perp:.4f} \n'
                     .format(epoch + 1, train_perp=train_perp, val_perp=val_perp))
        model.save_checkpoint(checkpoint_file % (epoch + 1))