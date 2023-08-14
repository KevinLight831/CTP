import argparse
import os,sys, copy
sys.path.append("..")
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.clip_pretrain import clip_pretrain
import utils
from data import create_dataset, create_sampler, create_loader
from product_evaluation import evaluation, itm_eval, evaluation_multi_modal
import codecs
import tensorboard_logger as tb_logger


sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
Eiters = 0

def init_train(model, data_loader, optimizer, epoch, device, config, iteration, lr_schedule, momentum_model):
    global Eiters
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = config['window_size']
    torch.cuda.empty_cache()  
    model.train()  
    data_loader.sampler.set_epoch(epoch)
    iters_per_epoch = len(data_loader)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[iters_per_epoch * epoch + i] 
        optimizer.zero_grad()
        id, image, caption = batch
        image = image.to(device,non_blocking=True)
        mode = 'CTP_init'
        loss_ita, loss_mlm= model.forward(mode, image, caption, iteration, epoch=0.9, momentum_model=momentum_model)  
        loss = (loss_ita + loss_mlm)
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb_logger.log_value('loss_ita', loss_ita, step=Eiters)
        tb_logger.log_value('loss_mlm', loss_mlm, step=Eiters)
        tb_logger.log_value('loss', loss, step=Eiters)
        Eiters+=1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def update_train(model, ref_model, data_loader, optimizer, epoch, device, config, iteration, lr_schedule, fisher, momentum_model):
    global Eiters
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita_dis', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm_dis', utils.SmoothedValue(window_size=config['window_size'], fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = config['window_size']
    torch.cuda.empty_cache()  
    model.train()  
    data_loader.sampler.set_epoch(epoch)
    iters_per_epoch = len(data_loader)


    if ref_model:
        ref_model.eval() 
        for param in ref_model.parameters():
            param.requires_grad = False

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = iters_per_epoch * epoch + i
        for j, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        optimizer.zero_grad()
        id, image, caption = batch
        image = image.to(device,non_blocking=True)
        mode = 'CTP'
        loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis = model.forward(mode, image, caption, iteration, epoch, ref_model, momentum_model=momentum_model)   
        loss = (loss_ita + loss_mlm) + (loss_ita_dis+ loss_mlm_dis)

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita_dis=loss_ita_dis.item())
        metric_logger.update(loss_mlm_dis=loss_mlm_dis.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb_logger.log_value('loss_ita', loss_ita.item(), step=Eiters)
        tb_logger.log_value('loss_mlm', loss_mlm.item(), step=Eiters)
        tb_logger.log_value('loss_ita_dis', loss_ita_dis.item(), step=Eiters)
        tb_logger.log_value('loss_mlm_dis', loss_mlm_dis.item(), step=Eiters)
        tb_logger.log_value('loss', loss.item(), step=Eiters)
        Eiters+=1

        torch.cuda.empty_cache()  
        del id, image, caption, loss, loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args,config,industry_id_label, all_id_info, device): 
    if 1:
        tb_logger.configure(os.path.join('./logger/tb_loggger'), flush_secs=5)
    crossmodal_dict = {}
    multimodal_dict = {}

    #### Model #### 
    print("Creating model")
    model = clip_pretrain(config=config, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device) 
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model._set_static_graph()
        model_without_ddp = model.module 

   #####task_list########
    train_list, test_list = [], []
    print("Start training")
    start_time = time.time()  
    for iteration , task_i in enumerate(config['task']):
        print(f"train task {iteration} : {task_i}")
        if iteration==0:
            if isinstance(task_i,list):    
                train_list = task_i
                test_list = task_i
            else:
                train_list=[task_i]
                test_list.append(task_i)
        else:
            test_list.append(task_i)
            if config['mode']== 'bound':
                train_list.append(task_i)
            else:
                train_list=[task_i]
            
        #### Dataset #### 
        print("Creating dataset")
        train_dataset = create_dataset('product_train', config, industry_id_label = industry_id_label, all_id_info=all_id_info, task_i_list=train_list)
        test_dataset = create_dataset('product_test', config, task_i_list=test_list)   
        datasets = [train_dataset,test_dataset]
        print('number of training samples: %d'%len(datasets[0]))
        print('number of testing samples: %d'%len(datasets[1]))
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)+ [None] 
        else:
            samplers = [None,None]    

        data_loader, test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size_train'], config['batch_size_test']], num_workers=[8,8], is_trains=[True,False], collate_fns=[None,None])  
        query_dataset, galley_dataset = create_dataset('product_query', config, task_i_list=test_list), create_dataset('product_gallery',config, task_i_list=test_list)  
        query_loader, gallery_loader = create_loader([query_dataset,galley_dataset],[None,None], batch_size=[128,128],num_workers=[8,8], is_trains=[False,False],collate_fns=[None,None]) 

        #### Train ####
        print("***** Running training *****")
        print(f"Num iters = {len(data_loader)},  Batch size = {config['batch_size_train']}")
        if iteration==0:
            momentum_model = copy.deepcopy(model_without_ddp)
            optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.module.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
            a_init_lr, b_min_lr = config['init_lr'], config['min_lr']
            print(f'now init_lr {a_init_lr}, now init_lr {b_min_lr}')
            lr_schedule = utils.cosine_scheduler(init_lr = a_init_lr, min_lr = b_min_lr, epochs =config['max_epoch'], niter_per_ep = len(data_loader))

            for epoch in range(0, config['max_epoch']):
                train_stats = init_train(model, data_loader, optimizer, epoch, device, config, iteration, lr_schedule,momentum_model)
                score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args, config)
                if utils.is_main_process():  
                    test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                    print(test_result)
                    txt_r1,img_r1,mean_r1,r_mean = test_result['txt_r1'],test_result['img_r1'],(test_result['txt_r1']+test_result['img_r1'])/2,test_result['r_mean']
                    crossmodal_dict[iteration] = '{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(txt_r1,img_r1,mean_r1,r_mean)
                    print(crossmodal_dict[iteration])  

        elif iteration>0:
            init_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            ref_model = copy.deepcopy(model_without_ddp)
            momentum_model = copy.deepcopy(model_without_ddp)
            model_params = {n : p for n, p in model.module.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW([{'params': model_params[key], 'lr':config['init_lr']} for key in model_params.keys()],weight_decay=config['weight_decay'])
            print(f'now init_lr {a_init_lr}, now min_lr {b_min_lr}')
            lr_schedule = utils.cosine_scheduler(init_lr = a_init_lr, min_lr = b_min_lr, epochs =config['max_epoch'], niter_per_ep = len(data_loader))

            model.module.load_state_dict(init_state_dict)
            for epoch in range(0, config['max_epoch']):
                train_stats = update_train(model, ref_model, data_loader, optimizer, epoch, device, config, iteration, lr_schedule, fisher=None, momentum_model=momentum_model)
                score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args, config)
                if utils.is_main_process():  
                    test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                    print(test_result)
                    txt_r1,img_r1,mean_r1,r_mean = test_result['txt_r1'],test_result['img_r1'],(test_result['txt_r1']+test_result['img_r1'])/2,test_result['r_mean']
                    crossmodal_dict[iteration] = '{:.2f},{:.2f},{:.2f},{:.2f},'.format(txt_r1,img_r1,mean_r1,r_mean)
                    print(crossmodal_dict[iteration]) 

        if utils.is_main_process():  
            map_result=evaluation_multi_modal(config, model_without_ddp, query_loader=query_loader,gallery_loader=gallery_loader,device=device)
            multimodal_dict[iteration] = '{:.2f}, {:.2f}, {:.2f}'.format(map_result['map1_vt'],map_result['map5_vt'],map_result['map10_vt'])
            for i in crossmodal_dict.keys():
                print(crossmodal_dict[i])
            for i in multimodal_dict.keys():
                print(multimodal_dict[i])

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_result.items()},  
                'task': task_i,
                'iteration': iteration,
                'txt_r1,img_r1,mean_r1,r_mean':  '{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(txt_r1,img_r1,mean_r1,r_mean),
            }                     
            save_model_name = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }
            if epoch==config['max_epoch'] -1:
                torch.save(save_model_name, os.path.join(args.output_dir, 'task_%02d.pth'%iteration)) 
                with open(os.path.join(args.output_dir, "log.json"),"a",encoding="utf-8") as f:
                    json.dump(log_stats,f,indent=2,ensure_ascii=False)

        dist.barrier()     

        torch.cuda.empty_cache()    
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--base_config', default='./configs/base.yaml')
    args = parser.parse_args()

    config_base = yaml.load(open(args.base_config, 'r',encoding='utf-8'), Loader=yaml.Loader)
    config_exp = yaml.load(open(args.config, 'r',encoding='utf-8'), Loader=yaml.Loader)

    config = utils.merge_data(config_base,config_exp)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(config)
        
    utils.init_distributed_mode(args)     
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False 
    file_json = open(os.path.join(args.output_dir, "log.json"),"w",encoding="utf-8").close()

    train_file = config['train_file']
    print('loading '+train_file)
    def read_json(file):
        f=open(file,"r",encoding="utf-8").read()
        return json.loads(f)
    industry_id_label = read_json(train_file)
    all_id_info={}
    for task_i in industry_id_label:
        for item_id, info in industry_id_label[task_i].items():
            all_id_info[item_id]=info
    print(len(all_id_info))

    main(args,config,industry_id_label, all_id_info, device)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'), allow_unicode=True)
