from pyexpat import features
import copy
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from datasets import load_from_disk
import io
from time import time
import pdb
from models.clip import *
from util import *


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = './model_output/'
    model_file_name = 'test_clip_entire_model_added_sigmoid_gradclip_maplayers_5.pt'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Initializing model")
    model = CLIPClassifier()
    model.to(device)

    print("Loading data")
    batch_size = 128
    combined = load_from_disk('./processed_data/combined_hateful_memes_dataset')
    train_data = combined['train']
    print(train_data)
    print('processing image...')
    train_dataset = CLIPProcessDataset(train_data)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # print("Sample item:", train_dataset[0])
    # print("Image size:", base64str_to_PILobj(train_dataset[0]['image']).size)
    dev_seen_data = combined['dev_seen']
    dev_seen_dataset = CLIPProcessDataset(dev_seen_data)
    dev_seen_loader = DataLoader(dev_seen_dataset, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    epochs = 15
    n_total_steps = len(train_dataloader)
    epo_loss = []
    for epoch in range(epochs):
        print("-----------------------------------")
        print("Epoch %d" % (epoch+1))
        print("-----------------------------------")
        start = time()
        batch_loss = []; total= 0 ; correct_normal=0
        for idx, batch in enumerate(train_dataloader):
            start_batch = time()
           
            labels = batch.pop("labels").squeeze(1).to(device)

            outputs = model(batch, device)
        
            logits = outputs
            predicted = torch.as_tensor((logits - 0.5) > 0, dtype=torch.int32)

            loss = model.cross_entropy_loss(logits.squeeze().float(), labels.float())

            optimizer.zero_grad()
            
            labels = labels.reshape(-1,1)
            total += labels.size(0)
            correct_normal += (predicted==labels).sum().item()
            if (idx+1)%5==0:
                # Print every 5 batches
                print(f"    Batch [{idx+1}/{n_total_steps}] Loss: {loss.item():.4f} Train Acc: {(correct_normal/total):.4f} ({(time()-start_batch):.4f}s)")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # add clipping
            optimizer.step()

            batch_loss.append(loss.item())


        train_acc = correct_normal/total

        dev_seen_acc = evaluate(model, dev_seen_loader, device=device)

        epo_loss.append(np.mean(batch_loss))
        print(f"Epoch {epoch+1} completed in {(time()-start):.2f}s. Loss={epo_loss[-1]:.4f}. train_acc={train_acc}. dev_seen_acc={dev_seen_acc} ({(time()-start):.4f}s)")
    
    
    torch.save(model, os.path.join(model_dir, model_file_name))
    print(f"Model saved to {os.path.join(model_dir, model_file_name)}")

# python train_clip.py >> clip.log 2>&1&