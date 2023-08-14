import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transform.randaugment import RandomAugment
from data.pretrain_product_dataset import pretrain_product, product_crossmodal_eval, product_multimodal_eval


def create_train_transforms(config, min_scale=0.5):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(
            min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_train


def create_test_transforms(config):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_test


def create_dataset(dataset, config, industry_id_label=None, all_id_info=None, task_i_list=[], memory_item_ids=[[], []], min_scale=0.2, ):
    transform_train = create_train_transforms(config, min_scale=min_scale)
    transform_test = create_test_transforms(config)

    if dataset == 'product_train':  
        train_dataset = pretrain_product(
            config, industry_id_label, all_id_info, transform_train, task_i_list, memory_item_ids)
        return train_dataset
    elif dataset == 'product_test':
        test_dataset = product_crossmodal_eval(
            config, transform_test, task_i_list)
        return test_dataset
    elif dataset == 'product_gallery':
        gallery_dataset = product_multimodal_eval(
            config, config['gallery_file'], config['gallery_image_root'], transform_test, task_i_list)
        return gallery_dataset
    elif dataset == 'product_query':
        query_dataset = product_multimodal_eval(
            config, config['query_file'], config['test_image_root'], transform_test, task_i_list)
        return query_dataset
    elif dataset == 'choose_exemplar':
        test_dataset = pretrain_product(
            config, industry_id_label, all_id_info, transform_test, task_i_list, memory_item_ids=[[], []])
        return test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
