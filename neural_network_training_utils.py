import torch
impport torch.nn as nn

class MeanSquareErrorLoss(nn.module):
    def forward(self, logits, targets):
        mse =nn.MSELoss()
        loss = mse(logits.squeeze(),targets)
        return loss

class BinaryCrossEntropyLoss(nn.Module):
    def forward(self,logits,targets):
        bce = nn.BCELoss()
        loss = bce(logits.squeeze(), targets)
        return loss

class Evaluator:
    def __init__(self):
        self.loss_func = nn.L1Loss()

    def evaluate(self, model, features, adj_pos,adj_neg, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(features, adj_ops, adj_neg)
        loss = self.loss_func(logits, labels)
        return loss, logits 

def prepare_data(data_dict,device):
    pos_adj = data_dict[].to(device).squeeze()
    neg_adg = data.dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict[].to(device).squeeze()
    mask = data_dict['mask']
    return pos_adj, neg_adj, features, labels, mask

def train_one_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = prepare_data(data,arags.device)
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], label[mask])
            loss.backward()
            optimizer.step()
            schedular.step()
            if batch_idx == 0:
                loss_return += loss.item()
    return loss_return / len(dataset_train)

def evaluate_one_epoch(args, model, dataset_eval, evaluate):
    tatol_loss = 0
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, feaetures, labels, mask = prepare_data(dara,args.device)
        loss, logits = evaluator.evaluate(model, features, pos_adj, neg_adj, neg_adj, labels, mask)
        break
    return loss.item(),logits 
    
        
                
            

    
