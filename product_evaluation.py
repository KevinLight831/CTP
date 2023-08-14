import torch
import utils
import torch.nn.functional as F
import time, datetime, os
import torch.distributed as dist
import numpy as np
import heapq
import json
from data import create_dataset, create_loader

def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

@torch.no_grad()
def evaluation(model, data_loader, device, args, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256 
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)

    #i2t
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 10000, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_i2t[start+i,topk_idx] = topk_sim

    #t2i    
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 10000, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)      
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:#list
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    

    test_list = [576,690,120,110,141,252,263,309,385]
    test_num = 0
    for i in range(len(test_list)):
        test_num+=test_list[i]
        if test_num>=len(ranks):
            tag = test_num - test_list[i]
            if tag==0:
                tag=1
            break

    tr_task0 = 100.0 * len(np.where(ranks[:tag] < 1)[0]) / len(ranks[:tag])
    tr_task1 = 100.0 * len(np.where(ranks[tag:] < 1)[0]) / len(ranks[tag:])
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)      

    
    ir_task0 = 100.0 * len(np.where(ranks[:tag] < 1)[0]) / len(ranks[:tag])
    ir_task1 = 100.0 * len(np.where(ranks[tag:] < 1)[0]) / len(ranks[tag:])
    print(f'task0 tr/ir{tr_task0:.2f}/{ir_task0:.2f}, task1 tr/ir{tr_task1:.2f}/{ir_task1:.2f}')  

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def compute_ap(rank_list,pos_set,topk):
    '''
        rank_list:
        pos_list:
        rank_list=["a","d","b","c"]
        pos_set=["b","c"]
        ap=compute_ap(rank_list,pos_set)
        print("ap: ",ap)
    '''
    intersect_size=0
    ap=0

    for i in range(topk):
        if rank_list[i] in pos_set:
            intersect_size += 1
            precision = intersect_size / (i+1)
            ap+=precision
    if intersect_size==0:
        return 0
    ap/=intersect_size

    return ap

@torch.no_grad()
def compute_gallery(model,data_loader,device):
    model.eval()
    item_ids = []
    vl_embeds = []
    for item_id, image, caption in data_loader:
        image = image.to(device)
        vl_embed = model.get_VL_feature(image,caption)
        vl_embed = F.normalize(vl_embed,dim=-1)
        vl_embeds.append(vl_embed)
        item_ids+=item_id
    vl_embeds = torch.vstack(vl_embeds)
    item_ids = np.hstack(item_ids)
    return vl_embeds, item_ids

@torch.no_grad()
def eval_gallery(score_matrix, query_ids, gallery_ids, query_id_label, gallery_id_label, gallery_label_id):
    max_topk = 10
    retrieval_results = []
    for q,each_score in zip(query_ids,score_matrix):
        max_index = heapq.nlargest(max_topk, range(len(each_score)), each_score.take)
        topk_item_id = gallery_ids[max_index]
        topk_item_id=[each_item_id for each_item_id in topk_item_id if each_item_id!=q]
        retrieval_results.append([q]+topk_item_id) 

    topk_list=[1,5,10]
    results={}
    for topk in topk_list:
        topk_temp=topk
        mAP, cnt = 0,0
        for index, rank_list in enumerate(retrieval_results):
            query_id=rank_list[0]
            rank_id_list=rank_list[1:]
            pos_set=[]
            cnt+=1
            query_labels=query_id_label[query_id]["label"]
            pos_set = gallery_label_id[query_labels]
            topk = min(topk_temp, len(pos_set),len(rank_id_list))
            ap=compute_ap(rank_id_list,pos_set,topk)
            mAP+=ap

        mAP=mAP/cnt*100

        results["top{}".format(topk_temp)]={
            "mAP": mAP,
        }
    return results


@torch.no_grad()
def evaluation_multi_modal(config, model, query_loader, gallery_loader, device):
    query_id_label, gallery_id_label, gallery_label_id = {},{},{}
    query_json, gallery_json = read_json(config['query_file']), read_json(config['gallery_file'])
    for item_id,info in gallery_json.items():
        label = info["cate_name"]
        gallery_id_label[item_id]={"label":label}
        if label not in gallery_label_id:
            gallery_label_id[label]=[item_id]
        else:
            gallery_label_id[label]+=[item_id]
    for item_id,info in query_json.items():
        label = info["cate_name"]
        query_id_label[item_id]={"label":label}
    start_time = time.time()
    query_vt_embed, query_item_id = compute_gallery(model,query_loader,device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('compute query time {}'.format(total_time_str)) 

    start_time = time.time()
    gallery_vt_embed, gallery_item_id = compute_gallery(model,gallery_loader,device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('compute gallery time {}'.format(total_time_str)) 
    print(f'query shape: {query_vt_embed.shape}, gallery shape: {gallery_vt_embed.shape}')

    sims_matrix_it2it = query_vt_embed @ gallery_vt_embed.t()


    start_time = time.time()
    result_vt = eval_gallery(sims_matrix_it2it.cpu().numpy(), query_item_id, gallery_item_id, query_id_label, gallery_id_label, gallery_label_id)
    print(f'reslut it2it: {result_vt}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    print('{:.2f}/{:.2f}/{:.2f}'.format(result_vt['top1']['mAP'],result_vt['top5']['mAP'],result_vt['top10']['mAP']))
    return {'map1_vt':result_vt['top1']['mAP'],'map5_vt':result_vt['top5']['mAP'], 'map10_vt':result_vt['top10']['mAP']}


def eval_all(args, config, device, model_without_ddp):
    results = {}
    results_map={}
    task_list=[]
    for iteration, task_i in enumerate(config['task']):
        task_list.append(task_i)
        print(task_i)
        test_dataset = create_dataset('product_test', config, task_i_list=task_list, min_scale=0.2)
        test_loader = create_loader([test_dataset],samplers=[None],batch_size=[config['batch_size_test']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
        checkpoint = torch.load(os.path.join(args.output_dir, 'task_%02d.pth'%iteration), map_location='cpu') 
        state_dict = checkpoint['model']    
        model_without_ddp.load_state_dict(state_dict,strict=False)

        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args, config)

        if utils.is_main_process():  
            results[iteration] = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
            print(results[iteration])

        query_dataset, galley_dataset = create_dataset('product_query', config, task_i_list=task_list), create_dataset('product_gallery',config, task_i_list=task_list)   
        query_loader, gallery_loader = create_loader([query_dataset,galley_dataset],[None,None], batch_size=[512,512],num_workers=[4,4], is_trains=[False,False],collate_fns=[None,None]) 
            
        results_map[iteration] = evaluation_multi_modal(config, model_without_ddp, query_loader=query_loader,gallery_loader=gallery_loader,device=device)

    for iteration, task_i in enumerate(config['task']):
        task_i_result = results[iteration]
        print(f'{iteration} {task_i}:{task_i_result}')
    for iteration, task_i in enumerate(config['task']):
        task_i_result = results[iteration]
        txt_r1,img_r1,mean_r1,r_mean = task_i_result['txt_r1'],task_i_result['img_r1'],(task_i_result['txt_r1']+task_i_result['img_r1'])/2,task_i_result['r_mean']
        print('{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(txt_r1,img_r1,mean_r1,r_mean)) 
    print('VT@map')
    for iteration, task_i in enumerate(config['task']): 
        print('{:.2f}/{:.2f}/{:.2f}'.format(results_map[iteration]['map1_vt'],results_map[iteration]['map5_vt'],results_map[iteration]['map10_vt'])) 
    








      
