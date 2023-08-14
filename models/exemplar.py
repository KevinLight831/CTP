import torch
import numpy as np
import faiss
import time

@torch.no_grad()
def compute_image_caption_features(model, data_loader, device):
    # test
    model.eval()    
    print('Computing features for exemplar...')
    image_feats = []
    text_feats = []
    id_lists = []
    for i, (id, image, text) in enumerate(data_loader): 
        image = image.to(device)
        image_feat, text_feat, image_embeds, image_atts, text, text_output = model.get_feature(image,text)  
        id_lists.append(id) 
        image_feats.append(image_feat.cpu())
        text_feats.append(text_feat.cpu())
    id_lists = np.hstack(id_lists)
    image_feats = torch.cat(image_feats,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    return id_lists, image_feats, text_feats

@torch.no_grad()
def compute_features(model, data_loader, device):
    # test
    model.eval()    
    print('Computing features for exemplar...')
    vl_feats = []
    id_lists = []
    for i, (id, image, text) in enumerate(data_loader): 
        image = image.to(device)
        vl_feat = model.get_VL_feature(image,text)  
        id_lists.append(id) 
        vl_feats.append(vl_feat.cpu())
    id_lists = np.hstack(id_lists)
    vl_feats = torch.cat(vl_feats,dim=0).numpy()
    return id_lists, vl_feats

def update_memory(item_id_list_np, mapped_prototypes, memory_size_each_task):
    D = mapped_prototypes.T
    D = D / np.linalg.norm(D, axis=0)

    mu = np.mean(D, axis=1)  
    alpha_dr_herding=np.zeros_like(item_id_list_np,dtype=np.float32)
    w_t = mu
    iter_herding = 0
    iter_herding_eff = 0

    while not (
            np.sum(alpha_dr_herding != 0) == min(memory_size_each_task, len(item_id_list_np))):
        tmp_t = np.dot(w_t, D)#The cosine distance from the center, the bigger, the closer
        ind_max = np.argmax(tmp_t)#index
        iter_herding_eff += 1
        if alpha_dr_herding[ind_max] == 0:
            alpha_dr_herding[ind_max] = 1 + iter_herding
            iter_herding += 1 #ind_max
        w_t = w_t + mu - D[:, ind_max]#mean shift

    alph=alpha_dr_herding
    alph = (alph > 0) * (alph < memory_size_each_task + 1) * 1.
    task_i_exemplar=item_id_list_np[np.where(alph==1)]
    task_i_exemplar_item_ids=task_i_exemplar.tolist()#[item_id for item_id in task_i_exemplar]

    return task_i_exemplar_item_ids

def kmeans_faiss(item_id_list_np, mapped_prototypes, memory_size_each_task):
    # faiss is used to quickly implement kmeans clustering
    D = mapped_prototypes / np.linalg.norm(mapped_prototypes, axis=1,keepdims=True)

    ncentroids = memory_size_each_task
    niter = 500
    verbose = False 
    d = D.shape[1]

    start_time = time.time()
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
    kmeans.cp.max_points_per_centroid = (D.shape[0] + ncentroids -1) // ncentroids
    kmeans.train(D)
 
    train_time = time.time()
    print(f'clustering time: {train_time - start_time} s')

    index = faiss.IndexFlatL2 (d)
    index.add (D)

    distance, index_list = index.search(kmeans.centroids, 1) #The vector index of the nearest cluster center

    task_i_exemplar=item_id_list_np[index_list.squeeze(-1)]
    task_i_exemplar_item_ids=[item_id for item_id in task_i_exemplar]
    return task_i_exemplar_item_ids
