import argparse
import os,sys
import ruamel.yaml as yaml
import time
import torch
from models.clip_pretrain import clip_pretrain
from data import create_dataset, create_sampler, create_loader
from product_evaluation import evaluation, itm_eval, evaluation_multi_modal
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/mnt2/save_1M_seq_finetune/4card_seq_CTP')        
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    time_a = time.time()
    args.config = os.path.join(args.output_dir,'config.yaml')
    config = yaml.load(open(args.config, 'r',encoding='utf-8'), Loader=yaml.Loader)      
    device = torch.device(args.device)

    print("Creating model")
    model = clip_pretrain(config=config, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                        vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)  
    print("Creating last task model")
    model_last = clip_pretrain(config=config, image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                        vit_ckpt_layer=config['vit_ckpt_layer'])
    model_last = model_last.to(device)  
    checkpoint_last = torch.load(os.path.join(args.output_dir, 'task_%02d.pth'%(len(config['task'])-1)), map_location='cpu') 
    state_dict_last = checkpoint_last['model']    
    model_last.load_state_dict(state_dict_last,strict=False)
    print("Creating dataset")
    task_list = []
    results = {}
    crossmodal_dict, crossmodal_dict_last = {}, {}
    multimodal_dict, multimodal_dict_last = {}, {}
    for iteration, task_i in enumerate(config['task']):
        task_list.append(task_i)
        print(task_i)
        # task_list= config['task']
        # iteration = 8
        test_dataset = create_dataset('product_test', config, task_i_list=task_list, min_scale=0.2)
        test_loader = create_loader([test_dataset],samplers=[None],batch_size=[256], num_workers=[8], is_trains=[False], collate_fns=[None])[0]

        query_dataset, galley_dataset = create_dataset('product_query', config, task_i_list=task_list), create_dataset('product_gallery',config, task_i_list=task_list)  
        query_loader, gallery_loader = create_loader([query_dataset,galley_dataset],[None,None], batch_size=[512,512],num_workers=[8,8], is_trains=[False,False],collate_fns=[None,None]) 

        checkpoint = torch.load(os.path.join(args.output_dir, 'task_%02d.pth'%iteration), map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict,strict=False)
        model_without_ddp = model
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args, config)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
        print(test_result)
        txt_r1,img_r1,mean_r1,r_mean = test_result['txt_r1'],test_result['img_r1'],(test_result['txt_r1']+test_result['img_r1'])/2,test_result['r_mean']
        crossmodal_dict[iteration] = [round(txt_r1,2), round(img_r1,2), round(mean_r1,2), round(r_mean,2)]
        print(crossmodal_dict[iteration]) 
        map_result=evaluation_multi_modal(config, model_without_ddp, query_loader=query_loader,gallery_loader=gallery_loader,device=device)
        multimodal_dict[iteration] = [round(map_result['map1_vt'],2), round(map_result['map5_vt'],2), round(map_result['map10_vt'],2)]
        #######eval last model#####
        print('=======test last model========')
        score_test_i2t, score_test_t2i = evaluation(model_last, test_loader, device, args, config)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
        print(test_result)
        txt_r1,img_r1,mean_r1,r_mean = test_result['txt_r1'],test_result['img_r1'],(test_result['txt_r1']+test_result['img_r1'])/2,test_result['r_mean']
        crossmodal_dict_last[iteration] = [round(txt_r1,2), round(img_r1,2), round(mean_r1,2), round(r_mean,2)]
        print(crossmodal_dict_last[iteration]) 
        map_result_last=evaluation_multi_modal(config, model_last, query_loader=query_loader,gallery_loader=gallery_loader,device=device)
        multimodal_dict_last[iteration] = [round(map_result_last['map1_vt'],2), round(map_result_last['map5_vt'],2), round(map_result_last['map10_vt'],2)]
        
    from prettytable import PrettyTable
    import numpy as np
    tb = PrettyTable(["Product Domain Index","txt_r1","img_r1","r1_m", "r_m", "mAP@1","mAP@5","mAP@10","BWT r_m","BWT mAP@1"])
    tb_last = PrettyTable(["Product Domain Index","txt_r1","img_r1","r1_m", "r_m", "mAP@1","mAP@5","mAP@10"])
    bwt_list = [[],[]]
    for i in crossmodal_dict.keys():
        bwt_list[0].append(crossmodal_dict[i][3]-crossmodal_dict_last[i][3])
        bwt_list[1].append(multimodal_dict[i][0]-multimodal_dict_last[i][0])
        if len(bwt_list[0])<=1:
            bwt_rm = '--'
            bwt_map1 = '--'
        else:
            bwt_rm = round(sum(bwt_list[0])/(len(bwt_list[0])-1),2)
            bwt_map1 = round(sum(bwt_list[1])/(len(bwt_list[1])-1),2)

        tb.add_row([i, crossmodal_dict[i][0], crossmodal_dict[i][1], crossmodal_dict[i][2], crossmodal_dict[i][3], multimodal_dict[i][0], multimodal_dict[i][1], multimodal_dict[i][2],bwt_rm, bwt_map1])
        tb_last.add_row([i, crossmodal_dict_last[i][0], crossmodal_dict_last[i][1], crossmodal_dict_last[i][2], crossmodal_dict_last[i][3], multimodal_dict_last[i][0], multimodal_dict_last[i][1], multimodal_dict_last[i][2]])
      
    print("The final model performance was tested in the continual task domain:")
    print(tb_last)
    print("The so-far model performance was tested in the continual task domain:")
    print(tb)
    print(f"time cost:{time.time()-time_a}s")


