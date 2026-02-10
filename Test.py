import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import GAMNet
from utils.dataloader import My_test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--pth_path', type=str, default='model_pth/EEMPVT/88EEMPVT-best.pth')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/UGASD_final')
    opt = parser.parse_args()
    model = GAMNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    ##### save_path #####
    save_path = '/root/autodl-tmp/pred/GAMNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_loader = My_test_dataset(opt.data_root, 352)
    num1 = test_loader.size
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        P1,P2 = model(image)
        res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        save = (res*255).astype(np.uint8)
        if not os.path.exists(os.path.join(save_path,name.split('/')[-2])):
            os.makedirs(os.path.join(save_path,name.split('/')[-2]))
        success = cv2.imwrite(os.path.join(save_path,name), save)
    print('UGASD Test', 'Finish!')
