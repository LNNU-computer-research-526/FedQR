import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from sklearn.cluster import KMeans
from utils import qr_2d_no_padding
import cvxpy as cp
##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone = True))
            
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params



def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        if ('fedlaw' in args.server_method) or ('fedawa' in args.server_method):
            client_params.append(client_nodes[idx].model.get_param(clone = True))
         
        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]

    return agg_weights, client_params

def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(prev_param))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params



global_T_weights_dict={}

def Server_update(args, central_node, client_nodes, select_list, size_weights,rounds_num=None,change=0):
    '''
    server update functions for baselines
    '''
    global size_weights_global
    global global_T_weights
    if rounds_num==change:
        size_weights_global=size_weights
    

    # receive the local models from clients
    if args.server_method == 'fedawa':
        agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights_global)
    else:
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
    print(agg_weights)
    

    if args.server_method == 'fedavg':
        avg_global_param = fedavg(client_params, agg_weights)
        
        central_node.model.load_state_dict(avg_global_param)
      
  

    elif args.server_method == 'fedawa':
        # print(rounds_num)

        if rounds_num==change:       
            global_T_weights=torch.tensor(agg_weights, dtype=torch.float32).to('cuda')

        
        avg_global_param,cur_global_T_weight = fedawa(args,client_params, agg_weights,central_node,rounds_num,global_T_weights)
        global_T_weights=cur_global_T_weight
        for i in range(len(select_list)):
            size_weights_global[select_list[i]] = global_T_weights[i]
        print("Global size weights:",size_weights_global)
        central_node.model.load_param(avg_global_param)

    elif args.server_method == 'fedqr':
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
        aggregated_params = fedqr(args, client_params, central_node, client_nodes, select_list)

        for idx in select_list:
            client_nodes[idx].model.load_state_dict(aggregated_params[idx])
        central_node.model.load_state_dict(aggregated_params[select_list[0]])
  



    else:
        raise ValueError('Undefined server method...')

    return central_node

#fedmy sample



# FedAvg
def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            # print(dict_local_params[name_param])
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        # print("value_global_param:",value_global_param)

        # print("name_param:"+name_param+':',fedavg_global_params[name_param]-value_global_param)


        # print("name_param:"+name_param+':',torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param)))
        # if name_param[-6:]=="weight":
        # a=1-torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param))
        # d.append(a.item())
        # d=0.999
        fedavg_global_params[name_param] = value_global_param
    # exit()
    # print(d)
    return fedavg_global_params









def unflatten_weight(M, flat_w):

    ws = (t.view(s) for (t, s) in zip(flat_w.split(M._weights_numels), M._weights_shapes))

    for (m, n), w in zip(M._weights_module_names, ws):
        # print(type(m))
        # exit()
        # print(m,n,w)
        if 'Batch' in str(type(m)):
            print(m,n,w)
        setattr(m, n, w)
    # exit()
    # yield
    # for m, n in M._weights_module_names:
    #     setattr(m, n, None)




def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)


        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            # print('cos_dis')
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            # print('euc_dis')
            C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        return C
#fedgroupavg_para group mean
def fedawa(args,parameters, list_nums_local_data,central_node,rounds,global_T_weight):
    param=central_node.model.get_param()

    global_params = copy.deepcopy(param)



    flat_w_list = [dict_local_params['flat_w'] for dict_local_params in parameters]



    local_param_list = torch.stack(flat_w_list)

    T_weights = to_var(global_T_weight)


    if args.server_optimizer=='sgd':
        Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif args.server_optimizer=='adam':
        Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))


    print("T_weights_before update:",torch.nn.functional.softmax(T_weights, dim=0))





    #num of server update

    for i in range(args.server_epochs):
        print("server weight update:",i)



        probability_train = torch.nn.functional.softmax(T_weights, dim=0)


        C = _cost_matrix(global_params['flat_w'].detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)

        reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
        print("reg_loss:",reg_loss)






        client_grad=local_param_list-global_params['flat_w']


        column_sum=torch.matmul(probability_train.unsqueeze(0),client_grad) #weighted sum


        # cosine sim
        # cos_sim = torch.nn.functional.cosine_similarity(client_grad.unsqueeze(0), column_sum.unsqueeze(1), dim=2)
        # print(cos_sim)
        #
        l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)


        # cosine sim
        # print("Cos_sim:",cos_sim)
        # sim_loss=-(torch.sum(probability_train*cos_sim, dim=(-2, -1)))
        #
        print("L2_distance:",l2_distance)
        sim_loss=(torch.sum(probability_train*l2_distance, dim=(-2, -1)))

        print("Sim_loss:",sim_loss)

        Loss=sim_loss+reg_loss
        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print("step "+str(i)+" Loss:"+str(Loss))



    global_T_weight=T_weights.data


    print("T_weights_after update:",global_T_weight)

    print("probability_train_after update:",probability_train)



    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]

    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, probability_train):
            # print(dict_local_params[name_param])
            list_values_param.append(dict_local_params[name_param] * num_local_data * args.gamma)
        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(probability_train)

        fedavg_global_params[name_param] = value_global_param

    return fedavg_global_params,global_T_weight


import copy
import torch
from utils import qr_2d_no_padding


def fedqr(args, client_params, central_node, client_nodes, select_list):
    if 'flat_w' in client_params[0]:
        device = client_params[0]['flat_w'].device
    else:
        first_param_key = next(iter(client_params[0].keys()))
        device = client_params[0][first_param_key].device

    parti_num = len(select_list)
    all_w = []
    all_q = []

    for i, idx in enumerate(select_list):
        fc_layer_name = 'linear.weight' if 'linear.weight' in client_params[i] else 'cls.weight'
        w = client_params[i][fc_layer_name].clone().to(device).to(torch.float32)
        q, _ = qr_2d_no_padding(w)
        all_w.append(w)
        all_q.append(q)

    Wavg = torch.mean(torch.stack(all_w), dim=0).to(device).to(torch.float32)

    weights_dict = cal_qr_weights(args, all_w, all_q, Wavg, parti_num, device)

    print("\n=== Client weight vectors ===")
    for client_idx in range(parti_num):
        actual_client_id = select_list[client_idx]
        weight_vector = weights_dict[client_idx]
        print(f"client{actual_client_id}weight vector：{weight_vector.tolist()}")


    aggregated_params = {}
    for client_idx in range(parti_num):
        current_id = select_list[client_idx]
        client_weights = weights_dict[client_idx]

        full_weights = [0.0] * parti_num
        full_weights[client_idx] = 1.0 / parti_num
        others_total_weight = 1.0 - full_weights[client_idx]

        other_idx = 0
        for i in range(parti_num):
            if i != client_idx and other_idx < len(client_weights):
                full_weights[i] = float(client_weights[other_idx] * others_total_weight)
                other_idx += 1

        weight_sum = sum(full_weights)
        full_weights = [w / weight_sum for w in full_weights]

        agg_params = copy.deepcopy(client_params[0])
        for key in agg_params:
            agg_params[key] = torch.zeros_like(agg_params[key], dtype=torch.float32, device=device)

        for i in range(parti_num):
            for key in agg_params:
                param = client_params[i][key].to(device).to(torch.float32)
                agg_params[key] += param * full_weights[i]

        aggregated_params[current_id] = agg_params

    return aggregated_params


def cal_qr_weights(args, all_w, all_q, Wavg, parti_num, device):
    weights_dict = {}
    Wavg_flat = Wavg.reshape(-1)
    lambda_reg = 0.01

    for current_id in range(parti_num):
        Q_star = all_q[current_id]
        Q_star_flat = Q_star.reshape(-1)
        Q_star_norm = torch.norm(Q_star_flat)

        other_ids = [i for i in range(parti_num) if i != current_id]
        n = len(other_ids)

        if n == 0:
            weights_dict[current_id] = np.array([])
            continue

        w_vectors = torch.stack([all_w[i] for i in other_ids])
        w_vectors_flat = w_vectors.reshape(n, -1)
        q_vectors = torch.stack([all_q[i] for i in other_ids])
        q_vectors_flat = q_vectors.reshape(n, -1)

        p_logits = nn.Parameter(torch.randn(n, device=device) * 0.1)

        if args.server_optimizer == 'sgd':
            optimizer = torch.optim.SGD([p_logits], lr=0.1, momentum=0.9)
        elif args.server_optimizer == 'adam':
            optimizer = torch.optim.Adam([p_logits], lr=0.01, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.SGD([p_logits], lr=0.1)

        for epoch in range(args.server_epochs):
            optimizer.zero_grad()

            p_prob = F.softmax(p_logits, dim=0)

            W_reconstructed = torch.matmul(p_prob, w_vectors_flat)
            reconstruction_loss = torch.sum((W_reconstructed - Wavg_flat) ** 2)

            q_norms = torch.norm(q_vectors_flat, dim=1)
            q_dot_Qstar = torch.matmul(q_vectors_flat, Q_star_flat)

            cosine_losses = torch.zeros(n, device=device)
            for i in range(n):
                if q_norms[i] > 1e-10 and Q_star_norm > 1e-10:
                    cosine_sim = q_dot_Qstar[i] / (q_norms[i] * Q_star_norm)
                    cosine_losses[i] = 1.0 - cosine_sim
                else:
                    cosine_losses[i] = 1.0

            similarity_loss = torch.sum(p_prob * cosine_losses)
            regularization_loss = lambda_reg * torch.sum(p_prob ** 2)
            total_loss = reconstruction_loss + similarity_loss + regularization_loss
            total_loss.backward()

            if epoch == 0:
                grad_norm = torch.norm(p_logits.grad) if p_logits.grad is not None else 0

            optimizer.step()

            if epoch % 10 == 0 or epoch == args.server_epochs - 1:
                with torch.no_grad():
                    p_current = F.softmax(p_logits, dim=0)
                    max_w = torch.max(p_current).item()
                    min_w = torch.min(p_current).item()

        with torch.no_grad():
            p_final = F.softmax(p_logits, dim=0)
            weights_np = p_final.cpu().numpy()
            weights_dict[current_id] = weights_np

            weight_sum = np.sum(weights_np)
            min_weight = np.min(weights_np)

    return weights_dict





