from functools import partial
from pyexpat import features
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from time import time
import pdb
import torch
import os
from util import evaluate, base64str_to_PILobj
from models.blip_itm import *


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = './model_output_test/'
    model_file_name = 'blip_entire_model.pt'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print(device)

    print("Initializing model")
    model = CustomBLIP()
    model.to(device)

    print("Loading data")
    batch_size = 16
    combined = load_from_disk('./processed_data/combined_hateful_memes_dataset')
    train_data = combined['train']
    print('processing image...')
    train_dataset = BLIPProcessDataset(train_data)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    dev_seen_data = combined['dev_seen']
    dev_seen_dataset = BLIPProcessDataset(dev_seen_data)
    dev_seen_loader = DataLoader(dev_seen_dataset, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)#torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    epochs = 20
    n_total_steps = len(train_dataloader)
    epo_loss = []
    for epoch in range(epochs):
        print("-----------------------------------")
        print("Epoch %d" % (epoch+1))
        print("-----------------------------------")
        start = time()
        batch_loss = [] ; total= 0 ; correct_normal=0
        for idx, batch in enumerate(train_dataloader):
            start_batch = time()
            
            labels = batch.pop("labels").squeeze(1).to(device)
            outputs = model(batch, device)
            logits = outputs
            predicted = torch.as_tensor((logits - 0.5) > 0, dtype=torch.int32)

            loss = model.cross_entropy_loss(logits.squeeze().float(), labels.float())

            labels = labels.reshape(-1,1)
            total += labels.size(0)
            correct_normal += (predicted==labels).sum().item()
           
            optimizer.zero_grad()
            if (idx+1)%5==0:
                print(f"    Batch [{idx+1}/{n_total_steps}] Loss: {loss.item():.4f} ({(time()-start_batch):.4f}s)")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            batch_loss.append(loss.item())

        train_acc = correct_normal/total
        dev_seen_acc = evaluate(model, dev_seen_loader, device=device)

        epo_loss.append(np.mean(batch_loss))
        print(f"Epoch {epoch+1} completed in {(time()-start):.2f}s. Loss={epo_loss[-1]:.4f}. train_acc={train_acc}. dev_seen_acc={dev_seen_acc} ({(time()-start):.4f}s)")

        torch.save(model, os.path.join(model_dir, model_file_name))
       
    
