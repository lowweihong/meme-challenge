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
from models._clip_unfreeze_block import *
from util import *


if __name__ == '__main__':
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    model_dir = './model_output/'
    # model_file_name = 'clip_entire_model_added_sigmoid_gradclip_laion-CLIP-ViT-B-32-laion2B-s34B-b79K-cross.pt'
    model_file_name = 'clip_entire_model_added_sigmoid_gradclip-cross-unfreeze-last-block.pt'
    print("Resize added")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Initializing model")
    model = CLIPClassifier()
    model.to(device)

    print("Loading data")
    batch_size = 8
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
    

    epochs = 20
    n_total_steps = len(train_dataloader)
    epo_loss = []
    labs = []
    preds = []
    best_auc = 0
    for epoch in range(epochs):
        model.train() # Tell the pytorch to train
        print("-----------------------------------")
        print("Epoch %d" % (epoch+1))
        print("-----------------------------------")
        start = time()
        batch_loss = []; total= 0 ; correct_normal=0
        best_acc = 0
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
            if (idx+1)%50==0:
                # Print every 5 batches
                print(f"    Batch [{idx+1}/{n_total_steps}] Loss: {loss.item():.4f} Train Acc: {((predicted==labels).sum().item()/labels.size(0)):.4f} ({(time()-start_batch):.4f}s)")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # add clipping
            optimizer.step()

            batch_loss.append(loss.item())
            labs.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
            preds.extend(logits.detach().cpu().numpy().reshape(-1).tolist())
            # print(preds, labs)
            # break
        # import pdb; pdb.set_trace()

        train_acc = correct_normal/total

        dev_seen_acc, auc = evaluate(model, dev_seen_loader, device=device)

        epo_loss.append(np.mean(batch_loss))
        print(f"Epoch {epoch+1} completed in {(time()-start):.2f}s. Loss={epo_loss[-1]:.4f}. train_acc={train_acc}. dev_seen_acc={dev_seen_acc}. dev_seen_auc={auc} ({(time()-start):.4f}s)")
    
        if auc > best_auc:
            torch.save(model, os.path.join(model_dir, model_file_name))
            print(f"Save checkpoint when epo={epoch}")
            best_auc = auc

    print(f"Completed. Final model saved to {os.path.join(model_dir, model_file_name)}")

# python train_clip.py >> clip.log 2>&1&