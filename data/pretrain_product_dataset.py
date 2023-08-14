from data.utils import pre_caption
import json

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def read_json(file):
    f = open(file, "r", encoding="utf-8").read()
    return json.loads(f)


class pretrain_product(Dataset):
    def __init__(self, config, industry_id_label, all_id_info, transform, task_i_list, memory_item_ids=[[], []]):
        self.image_path = config['train_image_root']
        self.max_words = config['max_words']
        self.transform = transform
        self.data_list = []
        self.industry_id_label = industry_id_label
        self.all_id_info = all_id_info

        for task_i in task_i_list:
            for item_id, info in self.industry_id_label[task_i].items():
                self.data_list.append(
                    (item_id, info["title"], info["cate_name"]))

        for i, item_id in enumerate(memory_item_ids[0]):
            label = memory_item_ids[1][i]
            self.data_list.append(
                (item_id, self.all_id_info[item_id]["title"], self.all_id_info[item_id]["cate_name"]))

        print("Total Pairs: {}".format(len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        item_id, caption, cate_name = self.data_list[index]
        image_path = "{}/{}.jpg".format(self.image_path, item_id)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
        image = self.transform(image)
        caption = pre_caption(caption, self.max_words)

        return item_id, image, caption


class product_crossmodal_eval(Dataset):
    def __init__(self, config, transform, task_i_list):

        self.all_id_info = read_json(config['test_file'])
        self.transform = transform
        self.image_root = config['test_image_root']
        self.max_words = config['max_words']

        self.data_list = []

        for item_id, info in self.all_id_info.items():
            industry_name = info["industry_name"]
            if industry_name not in task_i_list:
                continue
            else:
                self.data_list.append(
                    (item_id, info["title"],
                     info["cate_name"], info["industry_name"])
                )

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.dataset_len = len(self.data_list)
        print("Total Paire: {}".format(len(self.data_list)))

        for id, item in enumerate(self.data_list):
            self.image.append(item[0])
            self.img2txt[id] = [id]
            self.text.append(pre_caption(item[1], self.max_words))
            self.txt2img[id] = id

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        image_path = "{}/{}.jpg".format(self.image_root, self.image[index])

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
        image = self.transform(image)

        return image, index


class product_multimodal_eval(Dataset):
    def __init__(self, config, train_file, image_path, transform, task_i_list):
        self.train_file = train_file
        self.image_path = image_path
        self.max_words = config['max_words']
        self.transform = transform
        self.data_list = []

        print('loading '+self.train_file)
        self.all_id_info = read_json(self.train_file)

        for item_id, info in self.all_id_info.items():
            industry_name = info["industry_name"]
            if industry_name not in task_i_list:
                continue
            else:
                self.data_list.append(
                    (item_id, info["title"], info["cate_name"])
                )
        print("Total Paire: {}".format(len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        item_id, caption, cate_name = self.data_list[index]
        image_path = "{}/{}.jpg".format(self.image_path, item_id)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
        image = self.transform(image)
        caption = pre_caption(caption, self.max_words)

        return item_id, image, caption
