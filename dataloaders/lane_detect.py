import os
import numpy as np
from PIL import Image
from torch.utils import data

from dataloaders.utils import listFiles

class Lane_detect(data.Dataset):

    def __init__(self, root='path/to/datasets/lane_detect',n_classes=5, split="train", transform=None,extra=False):

        self.root = root
        self.split = split
        self.transform = transform
        self.files = {}
        self.n_classes = n_classes
        self.extra=extra

        self.images_path = os.path.join(self.root, self.split,'images')
        self.labels_path = os.path.join(self.root, self.split,'labels')            

        self.files[split] = listFiles(rootdir=self.images_path, suffix='.bmp')

        self.void_classes = [] #not to train
        self.valid_classes = [0,1,2,3,4]
        self.class_names = ['background','lane1','lane2','lane3','lane4']
        
        self.ignore_index = 25
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        #print(self.class_map)
        
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images.path))

        print("Found %d %s images" % (len(self.files[split]), split))
        
    
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        image_path = self.files[self.split][index].rstrip()
        label_path = os.path.join(self.labels_path,os.path.basename(image_path))
        _img = Image.open(image_path).convert('RGB')
        _tmp = np.array(Image.open(label_path).convert('L'), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)

        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
