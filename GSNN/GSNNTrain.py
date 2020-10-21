import torch
import numpy as np
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn import functional as F
import copy

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

class GSNNTrainer():
    def __init__(self, gsnn, optimizer, print_freq=10):
        self.gsnn = gsnn
        self.optimizer = optimizer
        self.print_freq = print_freq

    def train(self, adj, feature, label, train_mask, val_mask, test_mask, epochs, earlystopping=False):
        best_mean_acc = 0.0
        best_max_acc = 0.0
        max_acc = 0.0
        mean_acc = 0.0
        
        es = 100
        early_stopping = es

        y = label.clone()
        non_label = torch.cat((val_mask, test_mask))
        
        for epoch in range(epochs):
            self.gsnn.train()
            self.optimizer.zero_grad()
                
            y_pred_total_total, q_dis_total, y_total, y_encode = self.gsnn(feature, y, adj, non_label)
            reconstruct_total, kl_total, q_yl, kl_pq = self.getloss(y_pred_total_total, q_dis_total, y_total, y_encode, label, train_mask, non_label)
     
           
            
            lamda = 1.0
            loss = reconstruct_total + kl_total + kl_pq + lamda * q_yl

            loss.backward()
            self.optimizer.step()

            if earlystopping:
                max_acc, mean_acc, _, _ = self.evalGSNN(adj, feature, label, val_mask, 1, non_label)
                if mean_acc >= best_mean_acc:
                    best_mean_acc = mean_acc
                    best_model_para = copy.deepcopy(self.gsnn.state_dict())
                    early_stopping = es
                else:
                    early_stopping -= 1

            if epoch % self.print_freq == 0:
                print("Epoch: {}, Train_loss: {}.".format(epoch, loss.item()))
                print('-'*50)
            

            if early_stopping == 0:
                print('Early stopping!')
                break
        
        if earlystopping:
            self.gsnn.load_state_dict(best_model_para)   
        max_acc, mean_acc, y_p_max, y_p_mean = self.evalGSNN(adj, feature, label, test_mask, 1, non_label)
        print("Test_mean_acc: {}".format(mean_acc))
        print('-'*50)
        return mean_acc

    def getloss(self, y_pred_total_total, q_dis_total, y_total, y_encode, label, train_mask, non_label):
        # y_pred 
        label_scalar = torch.max(label, dim=1)[1]
        reconstruct_total = 0
        for j in range(len(y_pred_total_total)):
            y_pred_total = y_pred_total_total[j]
            for i in range(len(y_pred_total)):
                if i == 0:
                    reconstruct = F.cross_entropy(y_pred_total[0][train_mask], label_scalar[train_mask])
                else:
                    reconstruct += F.cross_entropy(y_pred_total[i][train_mask], label_scalar[train_mask])
            reconstruct_total += reconstruct / len(y_pred_total)
        reconstruct_total = reconstruct_total / len(y_pred_total_total)

        # kl
        prior_dis = Normal(torch.zeros(self.gsnn.get_z_dim()).cuda(), torch.ones(self.gsnn.get_z_dim()).cuda())
        kl_total = 0
        for j in range(len(y_pred_total_total)):
            kl_total += kl_divergence(q_dis_total[j], prior_dis).sum()
        kl_total = kl_total / len(y_pred_total_total)

        # q_yl
        label_scalar = torch.max(label, dim=1)[1]

        q_yl = F.cross_entropy(y_encode[train_mask], label_scalar[train_mask])

        # kl_pq
        kl_pq = 0
        for i in range(len(y_pred_total)):
            kl_pq += (F.gumbel_softmax(y_encode[non_label].detach(), tau=1.0, hard=True) * (-F.log_softmax(y_pred_total[i][non_label]))).sum()/len(non_label)
        kl_pq = kl_pq / len(y_pred_total)
        return reconstruct_total, kl_total, q_yl, kl_pq

    def evalGSNN(self, adj, feature, label, index_mask, sample_num, non_label):
        self.gsnn.eval()
        sample_list = []
        y_p_list = []
        total_size = len(index_mask)
        sample_total = torch.zeros_like(label, dtype=torch.float32)
        for _ in range(sample_num):
            y_eval, y_p = self.gsnn(feature, label, adj, non_label)
            sample_total = sample_total + y_eval
            
            y_eval_scalar = y_eval.argmax(dim = 1)
            y_p_scalar = y_p.argmax(dim = 1)
            label_scalar = label.argmax(dim = 1)
            sample_list.append((y_eval_scalar[index_mask]==label_scalar[index_mask]).sum().item())
            y_p_list.append((y_p_scalar[index_mask]==label_scalar[index_mask]).sum().item())

        sample_avg = sample_total / sample_num
        sample_scalar = sample_avg.argmax(dim = 1)

        return float(max(sample_list)) / total_size, float((sample_scalar[index_mask]==label_scalar[index_mask]).sum().item()) / total_size, float(max(y_p_list)) / total_size, float(np.mean(y_p_list)) / total_size

class GCNTrainer():
    def __init__(self, gcn, optimizer, print_freq=100):
        self.gcn = gcn
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.epoch_loss_history = []

    def train(self, adj, feature, label, train_mask, val_mask, test_mask, epochs, earlystopping=False):
        best_acc = 0.0
        acc = 0.0
        es = 100
        early_stopping = es

        for epoch in range(epochs):
            self.gcn.train()
            self.optimizer.zero_grad()

            y_pred = self.gcn(feature, adj)[train_mask]
            label_scalar = torch.max(label, dim=1)[1]
            label_scalar = label_scalar[train_mask]

            loss = F.cross_entropy(y_pred, label_scalar)
            loss.backward()
            self.optimizer.step()


            if epoch % self.print_freq == 0:
                print("Epoch: {}, Train_loss: {}, Val_acc: {}".format(epoch, loss.item(), acc))
                print('-'*50, early_stopping, best_acc)

            if earlystopping:    
                acc = self.evalGCN(adj, feature, label, val_mask)
                if acc >= best_acc:
                    best_acc = acc
                    best_model_para = copy.deepcopy(self.gcn.state_dict())
                    early_stopping = es
                else:
                    early_stopping -= 1
                
            if early_stopping == 0:
                print('Early stopping!')
                break
        
        #test early stopping
        if earlystopping:
            self.gcn.load_state_dict(best_model_para)
        acc = self.evalGCN(adj, feature, label, val_mask)
        print("Val_acc: {}".format(acc))
        acc = self.evalGCN(adj, feature, label, test_mask)
        print("Test_acc: {}".format(acc))
        print('-'*50)
        return acc


    def evalGCN(self, adj, feature, label, index_mask):
        self.gcn.eval()
        test_size = len(index_mask)
        y_pred = self.gcn(feature, adj)[index_mask]
        y_pred_scalar = y_pred.argmax(dim=1)
        label_scalar = label[index_mask].argmax(dim=1)

        return float((label_scalar==y_pred_scalar).sum().item()) / test_size
