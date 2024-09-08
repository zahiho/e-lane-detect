import os
import torch
import numpy as np
import matplotlib.pyplot as plt
def listFiles(rootdir='.', suffix='png'):

    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]
def get_lane_labels():
    return np.array([
         #[  0,   0,   0],
        [0, 0, 0],
        [100, 100, 100],
        [150, 150, 150],
        [200, 200, 200],
        [250, 250, 250]
        ])
def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks
def decode_segmap(label_mask, dataset, plot=False):

    if dataset == 'pascal':
      print()
    elif dataset == 'cityscapes':
        n_classes = 2
        label_colours = get_lane_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    
    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] =0
    
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r /255.0     
    rgb[:, :, 1] = g /255.0
    rgb[:, :, 2] = b /255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
def decode_segmap_cv(label_mask, dataset, plot=False):
    if dataset == 'pascal':
      print()
    elif dataset == 'lane':
        n_classes = 5
        label_colours = get_lane_labels()
    else:
        raise NotImplementedError
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] = 0
    rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
    rgb[:, :, 0] = b #/255.0     
    rgb[:, :, 1] = g #/255.0
    rgb[:, :, 2] = r #/255.0
    rgb = rgb.astype(np.uint8)
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()
def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


if __name__ == '__main__':
    print()
    ar=np.array([[0,7,10],[7,3,6]])
    