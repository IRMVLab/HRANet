import torch.utils.data as data
import numpy as np
from imageio import imread
from skimage.transform import resize as imresize
from path import Path
import os
import torch


def crawl_folders(folder, dataset='nyu'):
        imgs = []
        depths = []
        
        img_f = os.path.join(folder,'color')
        current_imgs = sorted(img_f.files('*.png'))
        
        img_d = os.path.join(folder,'depth')
        
        if dataset == 'nyu':
            current_depth = sorted((folder/'depth/').files('*.png'))
        elif dataset == 'kitti':
            current_depth = sorted(img_d.files('*.npy'))
        imgs.extend(current_imgs)
        depths.extend(current_depth)
        
        return imgs, depths

def load_tensor_image(img):
    #img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != 256 or w != 832):
        img = imresize(img, (256, 832)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img)/255-0.45)/0.225)
    return tensor_img
    
    
class TestSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, dataset='nyu'):
        self.root = Path(root)
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depth = crawl_folders(self.root, self.dataset)
        

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset=='nyu':
            depth = torch.from_numpy(imread(self.depth[index]).astype(np.float32)).float()/5000
        elif self.dataset=='kitti':
            depth = torch.from_numpy(np.load(self.depth[index]).astype(np.float32))

        if self.transform is not None:
            img = load_tensor_image(img)
            #img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)
