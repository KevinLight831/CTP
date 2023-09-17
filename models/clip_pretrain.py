from models.xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
from models.model_utils import create_vit
import torch.distributed as dist
import math

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma =nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) 

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class CLIP_Pretrain(nn.Module):
    def __init__(self, config,                
                 med_config = '../configs/albef_bert_chinese_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,   
                 mode=None  
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.config = config
        self.max_words = config['max_words']
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        if vit=='base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]     
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)      
               
        self.tokenizer = BertTokenizer.from_pretrained('/root/.cache/huggingface/bert-base-chinese') #Download ‘bert-base-chinese’ in advance and save it to a local path 
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width

        self.text_mlm_encoder = BertForMaskedLM.from_pretrained('/root/.cache/huggingface/bert-base-chinese',config=encoder_config)
        self.text_encoder = self.text_mlm_encoder.bert

        text_width = self.text_encoder.config.hidden_size
        
        if mode == 'LUCIR':
            self.vision_proj = CosineLinear(vision_width, embed_dim)
            self.text_proj = CosineLinear(text_width, embed_dim)
        else:
            self.vision_proj = nn.Linear(vision_width, embed_dim)
            self.text_proj = nn.Linear(text_width, embed_dim)
        
        self.temp = nn.Parameter(0.07*torch.ones([])).requires_grad_(False)  
        self.distill_temp = nn.Parameter(1.0*torch.ones([])).requires_grad_(False)
        self.mlm_probability = 0.15

        self.queue_size = 1024
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
   
    def get_raw_VL_feature(self, image,caption): #used in MAS
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = self.vision_proj(image_embeds[:,0,:])

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = self.text_proj(text_output.last_hidden_state[:,0,:])

        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,return_dict = True)
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        return image_feat, text_feat, fusion_out
        
    def get_raw_feature(self, image,caption):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feature = self.vision_proj(image_embeds[:,0,:]) 
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feature = self.text_proj(text_output.last_hidden_state[:,0,:])
        return image_feature, text_feature, image_embeds, image_atts, text, text_output

    def get_feature(self, image,caption):
        image_feature, text_feature, image_embeds, image_atts, text, text_output = self.get_raw_feature(image,caption)
        image_feat = F.normalize(image_feature,dim=-1) 
        text_feat = F.normalize(text_feature,dim=-1)
        return image_feat, text_feat, image_embeds, image_atts, text, text_output
       
    def get_VL_feature(self, image,caption): 
        #get the multimodal fusion feature
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)        
        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,return_dict = True)
        
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        fusion_out = F.normalize(fusion_out,dim=-1)
        return fusion_out

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def get_mlm_loss(self,text,image_embeds,image_atts ,device):        
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, len(self.tokenizer), device, targets=labels,
                                    probability_matrix = probability_matrix) 
        mlm_output = self.text_mlm_encoder(input_ids = input_ids, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    labels = labels
                                    ) 
        return mlm_output, input_ids, labels
    
    def distill_mlm(self, logit_mlm, ref_logits, labels):
        temp =self.distill_temp
        loss_mlm_dis = -torch.sum(F.log_softmax(logit_mlm/temp, dim=-1)*F.softmax(ref_logits/temp,dim=-1),dim=-1)
        loss_mlm_dis = loss_mlm_dis[labels!=-100].mean()
        return loss_mlm_dis
    
    def finetune_forward(self, image, caption):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)    
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm = mlm_output.loss

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2
        return loss_ita, loss_mlm


    def LWF_forward(self, image, caption, iteration, ref_model=None):
        raw_image_feature, raw_text_feature, image_embeds, image_atts, text, text_output = self.get_raw_feature(image,caption)

        raw_image_feat = F.normalize(raw_image_feature,dim=-1)  
        raw_text_feat = F.normalize(raw_text_feature,dim=-1)  

        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm_new = mlm_output.loss
        loss_mlm = loss_mlm_new 

        loss_ita, loss_ita_dis, loss_mlm_dis = 0*loss_mlm, 0*loss_mlm, 0*loss_mlm

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2

        if iteration >0:
            with torch.no_grad():
                ref_image_feature, ref_text_feature, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_raw_feature(image,caption)
                ref_mlm_output = ref_model.text_mlm_encoder(input_ids = input_ids_new, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = ref_image_embeds,
                                       encoder_attention_mask = ref_image_atts,      
                                       return_dict = True,
                                       labels = labels_new,   
                                      ) 

            loss_i_dis = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(raw_image_feature/self.temp, dim=1),F.softmax(ref_image_feature/self.temp, dim=1))
            loss_t_dis = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(raw_text_feature/self.temp, dim=1),F.softmax(ref_text_feature/self.temp, dim=1))
            loss_ita_dis  = (loss_i_dis + loss_t_dis) /2

            loss_mlm_dis = self.distill_mlm(mlm_output.logits,ref_mlm_output.logits, labels_new) 

        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

    def LUCIR_forward(self, image, caption, iteration, ref_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)   
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm_new = mlm_output.loss
        loss_mlm = loss_mlm_new 

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2
        loss_ita_dis, loss_mlm_dis = 0*loss_ita, 0*loss_ita

        if iteration >0:
            with torch.no_grad():
                ref_image_feat, ref_text_feat, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_feature(image,caption)
                ref_mlm_output = ref_model.text_mlm_encoder(input_ids = input_ids_new, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = ref_image_embeds,
                                       encoder_attention_mask = ref_image_atts,      
                                       return_dict = True,
                                       labels = labels_new,   
                                      ) 

            loss_cos_i = 1- torch.cosine_similarity(F.normalize(image_embeds,dim=-1), F.normalize(ref_image_embeds,dim=-1)).mean() 
            loss_cos_t = 1- torch.cosine_similarity(F.normalize(text_output.last_hidden_state,dim=-1), F.normalize(ref_text_output.last_hidden_state,dim=-1)).mean()  

            loss_cos = (loss_cos_i+ loss_cos_t)/2
            loss_ita_dis += loss_cos
            loss_mlm_dis = self.distill_mlm(mlm_output.logits,ref_mlm_output.logits,labels_new ) 
        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

        
    def CTP_init(self, image, caption, momentum_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm = mlm_output.loss 

        with torch.no_grad():
            model_pairs = [[self.visual_encoder,momentum_model.visual_encoder],
                        [self.vision_proj,momentum_model.vision_proj],
                        [self.text_mlm_encoder,momentum_model.text_mlm_encoder],
                        [self.text_proj,momentum_model.text_proj],
                    ]
            self._momentum_update(model_pairs, momentum=0.995)
            image_feat_m, text_feat_m, image_embeds_m, image_atts_m, text_m, text_output_m = momentum_model.get_feature(image,caption) 
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)              
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            mlm_output_m = momentum_model.text_mlm_encoder(input_ids = input_ids_new, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds_m,
                                    encoder_attention_mask = image_atts_m,      
                                    return_dict = True,
                                    labels = labels_new,   
                                    )
     
        sim_i2t_md = raw_image_feat @ text_feat_all / self.temp
        sim_t2i_md = raw_text_feat @ image_feat_all / self.temp

        sim_targets = torch.zeros(sim_i2t_md.size()).to(image.device)
        sim_targets.fill_diagonal_(1)          

        loss_i2t_md = -torch.sum(F.log_softmax(sim_i2t_md, dim=1)*sim_targets,dim=1).mean()
        loss_t2i_md = -torch.sum(F.log_softmax(sim_t2i_md, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t_md+loss_t2i_md)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m) 

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita += (loss_i2t+loss_t2i)/2
        loss_mlm += self.distill_mlm(mlm_output.logits, mlm_output_m.logits, labels_new)
        return loss_ita, loss_mlm

    def CTP(self, image, caption, ref_model, momentum_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm = mlm_output.loss 

        loss_ita_dis, loss_mlm_dis = 0*loss_mlm, 0*loss_mlm

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2

        mask_index = torch.arange(batch_size * dist.get_rank(), batch_size * (dist.get_rank()+1)).unsqueeze_(-1).to(image.device)

        sim_i2i = (raw_image_feat @ all_gather_with_grad(raw_image_feat).T).scatter(1,mask_index,-1000)
        sim_t2t = (raw_text_feat @ all_gather_with_grad(raw_text_feat).T).scatter(1,mask_index,-1000)

        with torch.no_grad():
            model_pairs = [[self.visual_encoder,ref_model.visual_encoder,momentum_model.visual_encoder],
                        [self.vision_proj,ref_model.vision_proj,momentum_model.vision_proj],
                        [self.text_mlm_encoder,ref_model.text_mlm_encoder,momentum_model.text_mlm_encoder],
                        [self.text_proj,ref_model.text_proj,momentum_model.text_proj],
                    ]
            self._momentum_update_three(model_pairs,momentum=0.9)
            image_feat_m, text_feat_m, image_embeds_m, image_atts_m, text_m, text_output_m = momentum_model.get_feature(image,caption) 
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)              
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            
            mlm_output_m = momentum_model.text_mlm_encoder(input_ids = input_ids_new, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds_m,
                                    encoder_attention_mask = image_atts_m,      
                                    return_dict = True,
                                    labels = labels_new,   
                                    )

            sim_i2t_mom = image_feat_m @ text_feat_all / self.temp  
            sim_targets = torch.zeros(sim_i2t_mom.size()).to(image.device)
            sim_targets.fill_diagonal_(1)     
   
        sim_i2t_md = raw_image_feat @ text_feat_all / self.temp
        sim_t2i_md = raw_text_feat @ image_feat_all / self.temp
        loss_i2t_md = -torch.sum(F.log_softmax(sim_i2t_md, dim=1)*sim_targets,dim=1).mean()
        loss_t2i_md = -torch.sum(F.log_softmax(sim_t2i_md, dim=1)*sim_targets,dim=1).mean() 

        loss_ita_dis += (loss_i2t_md+loss_t2i_md)/2
        loss_mlm_dis += self.distill_mlm(mlm_output.logits,mlm_output_m.logits, labels_new)

        self._dequeue_and_enqueue(image_feat_m, text_feat_m) 

        with torch.no_grad():
            ref_image_feat, ref_text_feat, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_feature(image,caption)
            sim_i2t_ref = (ref_image_feat @ concat_all_gather(ref_text_feat).T)
            sim_t2i_ref = (ref_text_feat @ concat_all_gather(ref_image_feat).T)

            sim_i2i_ref = (ref_image_feat @ concat_all_gather(ref_image_feat).T).scatter(1,mask_index,-1000)
            sim_t2t_ref = (ref_text_feat @ concat_all_gather(ref_text_feat).T).scatter(1,mask_index,-1000)
  
        #cross modal
        loss_ita_dis_i2t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2t/self.temp, dim=1),F.softmax(sim_i2t_ref/self.temp, dim=1))
        loss_ita_dis_t2i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2i/self.temp, dim=1),F.softmax(sim_t2i_ref/self.temp, dim=1))
        loss_ita_dis += (loss_ita_dis_i2t + loss_ita_dis_t2i) /2
        #same modal
        loss_cos_i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2i/self.temp, dim=1),F.softmax(sim_i2i_ref/self.temp, dim=1))
        loss_cos_t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2t/self.temp, dim=1),F.softmax(sim_t2t_ref/self.temp, dim=1))
        loss_ita_dis += (loss_cos_i + loss_cos_t) /2
               
        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

   
    def forward(self, mode, image, caption, iteration=0, epoch=0, ref_model=None, momentum_model=None):

        if mode == 'finetune': 
            loss_ita, loss_mlm= self.finetune_forward(image, caption)
            return loss_ita, loss_mlm
        elif mode == 'LWF':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis = self.LWF_forward(image, caption, iteration, ref_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis   
        elif mode == 'LUCIR':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis= self.LUCIR_forward(image, caption, iteration, ref_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis
        elif mode == 'CTP':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis= self.CTP(image, caption, ref_model, momentum_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis
        elif mode == 'CTP_init':
            loss_ita, loss_mlm= self.CTP_init(image, caption, momentum_model)
            return loss_ita, loss_mlm 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self,model_pairs, momentum):
        for model_pair in model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * momentum + param.data * (1. - momentum)

    @torch.no_grad()        
    def _momentum_update_three(self,model_pairs, momentum):
        for model_pair in model_pairs:           
            for param, param_r, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters(), model_pair[2].parameters()):
                param_m.data = param_m.data * momentum + param.data * (1. - momentum)/2 + param_r.data * (1. - momentum)/2


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


def clip_pretrain(**kwargs):
    model = CLIP_Pretrain(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     
    
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
