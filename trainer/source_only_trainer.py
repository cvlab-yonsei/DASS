import torch
from utils.optimize import *
from .base_trainer import BaseTrainer
#from pytorch_memlab import profile
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import  torch.optim as optim
from tqdm import tqdm
import neptune
import math
from PIL import Image
import copy
import torch.nn.functional as F

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def iter(self, batch):
        img, seg_label, _, _, name = batch
        seg_label = seg_label.long().to(self.device)
        b, c, h, w = img.shape
        seg_pred = self.model(img.to(self.device))
        seg_loss = F.cross_entropy(seg_pred, seg_label, ignore_index=255)
        self.losses.seg_loss = seg_loss
        loss = seg_loss
        loss.backward()

    def train(self):
        if self.config.neptune:
            neptune.init(project_qualified_name="leegeon30/segmentation-DA")
            neptune.create_experiment(params=self.config, name=self.config['note'])

        if self.config.multigpu:
            self.optim = optim.SGD(self.model.module.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optim = optim.SGD(self.model.optim_parameters(self.config.learning_rate),
                          lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        self.loader, _ = dataset.init_source_dataset(self.config)#, source_list=self.config.src_list)

        best_miou = 0
        cu_iter = 0
        print(len(self.loader))
        for i_iter, batch in enumerate(self.loader):
            cu_iter +=1
            adjust_learning_rate(self.optim, cu_iter, self.config)
            self.optim.zero_grad()
            self.losses = edict({})
            losses = self.iter(batch)

            self.optim.step()
            if cu_iter % self.config.print_freq ==0:
                self.print_loss(cu_iter)
            if self.config.val and cu_iter % self.config.val_freq ==0 and cu_iter!=0:
                miou = self.validate()
                if best_miou < miou:
                    best_miou = miou
                    self.save_model('best_source_only')
                self.model = self.model.train()
        if self.config.neptune:
            neptune.stop()

    def resume(self):
        self.tea = copy.deepcopy(self.model)
        self.round_start = self.config.round_start #int(math.ceil(iter_num/self.config.num_steps) -1 )
        print('Resume from Round {}'.format(self.round_start))
        if self.config.lr_decay == 'sqrt':
            self.config.learning_rate = self.config.learning_rate/((math.sqrt(2))**self.round_start)

    def save_best(self, name):
        name = str(name)
        if 'pth' not in name:
            name = name +'.pth'
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], name))

    def save_model(self, iter, rep_teacher=False):
        tmp_name = '_'.join(("Synthia", str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))

    def validate(self):
        self.model = self.model.eval()
        testloader = dataset.init_test_dataset(self.config, self.config.target, set='val')
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        union = torch.zeros(19, 1,dtype=torch.float).cuda().float()
        inter = torch.zeros(19, 1, dtype=torch.float).cuda().float()
        preds = torch.zeros(19, 1, dtype=torch.float).cuda().float()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                output =  self.model(image.cuda())
                label = label.cuda()
                output = interp(output).squeeze()
                #output = output.squeeze()
                C, H, W = output.shape
                Mask = (label.squeeze())<C

                pred_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                pred_e = pred_e.repeat(1, H, W).cuda()
                pred = output.argmax(dim=0).float()
                pred_mask = torch.eq(pred_e, pred).byte()
                pred_mask = pred_mask*Mask.byte()

                label_e = torch.linspace(0,C-1, steps=C).view(C, 1, 1)
                label_e = label_e.repeat(1, H, W).cuda()
                label = label.view(1, H, W)
                label_mask = torch.eq(label_e, label.float()).byte()
                label_mask = label_mask*Mask.byte()

                tmp_inter = label_mask+pred_mask.byte()
                cu_inter = (tmp_inter==2).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_union = (tmp_inter>0).view(C, -1).sum(dim=1, keepdim=True).float()
                cu_preds = pred_mask.view(C, -1).sum(dim=1, keepdim=True).float()
                union+=cu_union
                inter+=cu_inter
                preds+=cu_preds

            iou = inter/union
            acc = inter/preds
            if self.config.source=='synthia':
                iou = iou.squeeze()
                cass13_iou = torch.cat((iou[:3], iou[6:9], iou[10:14], iou[15:16], iou[17:]))
                class13_miou = class13_iou.mean().item()
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
                print(class13_iou)
                print(class13_miou)
            mIoU = iou.mean().item()
            mAcc = acc.mean().item()
            iou = iou.cpu().numpy()
            #print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            if self.config.neptune:
                neptune.send_metric('mIoU', mIoU)
                neptune.send_metric('mAcc', mAcc)
        return class13_miou

    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                neptune.send_metric(key, self.losses[key].item())
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)
