import torch
from utils.optimize import adjust_learning_rate
from .base_trainer import BaseTrainer
from utils.flatwhite import *
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import neptune
import math
from PIL import Image
from utils.meters import AverageMeter, GroupAverageMeter
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import operator
import pickle
import random
from utils.kmeans import kmeans_cluster
from utils.func import Acc, thres_cb_plabel, gene_plabel_prop, mask_fusion
from utils.pool import Pool
from utils.flatwhite import *
from trainer.base_trainer import *

class Trainer(BaseTrainer):
    def __init__(self, model, config, writer):
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        self.model = model
        self.config = config
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def entropy_loss(self, p):
        p = F.softmax(p, dim=1)
        log_p = F.log_softmax(p, dim=1)
        loss = -torch.sum(p * log_p, dim=1)
        return loss

    def metric_loss(self, feature_s, label_s, feature_t, label_t):
        _, ch, feature_s_h, feature_s_w = feature_s.size()
        label_s_resize = F.interpolate(label_s.float().unsqueeze(0), size=(feature_s_h, feature_s_w))

        _, _, feature_t_h, feature_t_w = feature_t.size()
        label_t_resize = F.interpolate(label_t.float().unsqueeze(0), size=(feature_t_h, feature_t_w))

        source_list = torch.unique(label_s_resize.float())
        target_list = torch.unique(label_t_resize.float())

        overlap_classes = []
        for i, index in enumerate(torch.unique(label_s_resize)):
            if index in torch.unique(label_t_resize) and index != 255.:
                overlap_classes.append(index.detach())

        source_prototypes = torch.zeros(size=(19, ch)).cuda()
        for i, index in enumerate(source_list):
            if index!=255.:
                fg_mask_s = ((label_s_resize==index)*1).cuda().detach()
                prototype_s = (fg_mask_s*feature_s).squeeze().resize(ch,feature_s_h*feature_s_w).sum(-1)/fg_mask_s.sum()
                source_prototypes[int(index)] = prototype_s

        cs_map = torch.matmul(F.normalize(source_prototypes, dim = 1), F.normalize(feature_t.squeeze().resize(ch,feature_t_h*feature_t_w), dim=0))
        cs_map[cs_map==0] = -1
        cosine_similarity_map = F.interpolate(cs_map.resize(1, 19,feature_t_h, feature_t_w), size=label_t.size()[-2:])
        cosine_similarity_map *= 10

        cross_entropy_weight = torch.zeros(size=(19,1))
        for i, element in enumerate(overlap_classes):
            cross_entropy_weight[int(element)] = 1

        cross_entropy_weight = cross_entropy_weight.cuda()
        prototype_loss = torch.nn.CrossEntropyLoss(weight= cross_entropy_weight, ignore_index=255)

        prediction_by_cs = F.softmax(cosine_similarity_map, dim=1)
        target_predicted = prediction_by_cs.argmax(dim=1)
        confidence_of_target_predicted = target_predicted.max(dim=1).values
        confidence_mask  = (confidence_of_target_predicted>0.8)*1
        target_predicted[target_predicted==0] = 20
        masked_target_predicted = target_predicted * confidence_mask
        masked_target_predicted[masked_target_predicted==0] = 255
        masked_target_predicted[masked_target_predicted==20] = 0
        masked_target_predicted_resize = F.interpolate(masked_target_predicted.float().unsqueeze(0), size=(feature_t_h, feature_t_w), mode='nearest')

        label_t_resize_new = label_t_resize.clone().contiguous()
        label_t_resize_new[label_t_resize_new==255] = masked_target_predicted_resize[label_t_resize_new==255]

        target_list2 = torch.unique(label_t_resize_new)

        overlap_classes2 = []
        for i, index in enumerate(torch.unique(label_s_resize)):
            if index in torch.unique(label_t_resize_new) and index != 255.:
                overlap_classes2.append(index.detach())

        target_prototypes = torch.zeros(size=(19, ch)).cuda()
        for i, index in enumerate(target_list):
            if index!=255.:
                #fg_mask_t = ((label_t_resize_new==index)*1).cuda()
                fg_mask_t = ((label_t_resize==index)*1).cuda().detach()
                prototype_t = (fg_mask_t*feature_t).squeeze().resize(ch,feature_t_h*feature_t_w).sum(-1)/fg_mask_t.sum()
                target_prototypes[int(index)] = prototype_t

        cs_map2 = torch.matmul(F.normalize(target_prototypes, dim = 1), F.normalize(feature_s.squeeze().resize(ch,feature_s_h*feature_s_w), dim=0))
        cs_map2[cs_map2==0] = -1
        cosine_similarity_map2 = F.interpolate(cs_map2.resize(1, 19, feature_s_h, feature_s_w), size=label_s.size()[-2:])
        cosine_similarity_map2 *= 10

        prototype_loss2 = torch.nn.CrossEntropyLoss(ignore_index=255)

        metric_loss1 = prototype_loss(cosine_similarity_map, label_t)
        metric_loss2 = prototype_loss(cosine_similarity_map2, label_s)

        metric_loss = self.config.lamb_metric1 * metric_loss1 + self.config.lamb_metric2 * metric_loss2
        return metric_loss

    def iter(self, source_batch, target_batch):
        img_s, label_s, _, _, name = source_batch
        img_t, label_t, _, _, name = target_batch

        #pred_s, feature_s = self.model.forward(img_s.cuda())
        pred_s, feature_s = self.model.forward(img_s.cuda(), source=True)
        #pred_t, feature_t = self.model.forward(img_t.cuda())
        pred_t, feature_t = self.model.forward(img_t.cuda(), source=False)

        label_s = label_s.long().to(self.device)
        label_t = label_t.long().to(self.device)

        label_s_0 = label_s[0,:,:].unsqueeze(0).contiguous()
        label_s_1 = label_s[1,:,:].unsqueeze(0).contiguous()
        feature_s_0 = feature_s[0,:,:,:].unsqueeze(0).contiguous()
        feature_s_1 = feature_s[1,:,:,:].unsqueeze(0).contiguous()

        label_t_0 = label_t[0,:,:].unsqueeze(0).contiguous()
        label_t_1 = label_t[1,:,:].unsqueeze(0).contiguous()
        feature_t_0 = feature_t[0,:,:,:].unsqueeze(0).contiguous()
        feature_t_1 = feature_t[1,:,:,:].unsqueeze(0).contiguous()

        loss_s = F.cross_entropy(pred_s, label_s, ignore_index=255)
        loss_t = F.cross_entropy(pred_t, label_t, ignore_index=255)
        loss_seg = (loss_s + self.config.lambt * loss_t)

        loss_e = self.entropy_loss(pred_s).mean() + self.config.lambt*self.entropy_loss(pred_t).mean()
        loss_e = self.config.lamb * loss_e

        loss_metric_0 = self.metric_loss(feature_s_0, label_s_0, feature_t_0, label_t_0)
        loss_metric_1 = self.metric_loss(feature_s_1, label_s_1, feature_t_1, label_t_1)
        #loss_metric_2 = self.metric_loss(feature_s_0, label_s_0, feature_t_1, label_t_1)
        #loss_metric_3 = self.metric_loss(feature_s_1, label_s_1, feature_t_0, label_t_0)
        loss_metric = (loss_metric_0 + loss_metric_1 ) / 2

        loss = loss_seg + loss_e + loss_metric

        self.losses.loss_source = loss_s
        self.losses.loss_target = loss_t
        self.losses.loss_entropy = loss_e
        self.losses.loss_metric = loss_metric

        loss.backward()

    def train(self):
        if self.config.neptune:
            neptune.init(project_qualified_name="leegeon30/segmentation-DA")
            neptune.create_experiment(params=self.config, name=self.config["note"],
                                      upload_source_files=['run.py', 'trainer/*.py', 'model/DeeplabV2.py'])
        if self.config.resume:
            self.resume()
        else:
            self.round_start = 0
        best_miou = 0
        best_r = 0
        best_epoch = 0
        best_iter = 0

        for r in range(self.round_start, self.config.round):
            self.model = self.model.train()
            #self.source_all = get_list(self.config.gta5.data_list)
            self.target_all = get_list(self.config.cityscapes.data_list)#[:50]

            if r!=0:
                self.cb_thres = self.gene_thres(self.config.cb_prop + self.config.thres_inc * r)
                self.save_pred(r)
            self.plabel_path = osp.join(self.config.plabel, self.config.note, str(r))

            self.optim = torch.optim.SGD(
                self.model.optim_parameters(self.config.learning_rate),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

            self.source_loader, self.target_loader, _ = dataset.init_pair_dataset(
                self.config,
                plabel_path=self.plabel_path,
                source_plabel_path = None,
                target_selected=self.target_all)

            self.target_loader_iter = iter(self.target_loader)
            for epoch in range(self.config.epochs):
                for i_iter, source_batch in tqdm(enumerate(self.source_loader)):
                    self.model.train()
                    if i_iter % (len(self.target_loader.dataset) // self.config.batch_size - 1) == 0:
                        self.target_loader_iter = iter(self.target_loader)
                    target_batch = self.target_loader_iter.next()
                    cu_step = epoch * self.config.num_steps + i_iter
                    self.losses = edict({})
                    self.optim.zero_grad()
                    adjust_learning_rate(self.optim, cu_step, self.config)
                    self.iter(source_batch, target_batch)

                    self.optim.step()
                    if i_iter % self.config.print_freq ==0:
                        self.print_loss(i_iter)
                    if i_iter % self.config.val_freq ==0 and i_iter != 0:
                        miou = self.validate()
                        if best_miou < miou:
                            best_miou = miou
                            best_r = r
                            best_epoch = epoch
                            best_iter = i_iter
                            print('best miou : %.2f, r : %d, epoch : %d, iter: %d'%(best_miou*100, best_r, best_epoch, best_iter))
                            self.save_model('baseline')
                    #if i_iter % self.config.save_freq ==0 and i_iter!=0:
                    #    self.save_model(r*self.config.num_steps + cu_step)
                    if i_iter > self.config.num_steps:
                        break
                miou = self.validate()
            #self.config.cb_prop += 0.05
            #self.config.learning_rate = self.config.learning_rate / math.sqrt(2)
            self.config.learning_rate = self.config.learning_rate / 2
        if  self.config.neptune:
            neptune.stop()

    def resume(self):
        iter_num = self.config.init_weight[-5]#.split(".")[0].split("_")[1]
        iter_num = int(iter_num)
        self.round_start = int(math.ceil((iter_num+1) / self.config.epochs) )
        print("Resume from Round {}".format(self.round_start))
        if self.config.lr_decay == "sqrt":
            self.config.learning_rate = self.config.learning_rate / (
                (math.sqrt(2)) ** self.round_start
            )

    def gene_thres(self, prop, num_cls=19):
        print('[Calculate Threshold using config.cb_prop]') # r in section 3.3

        probs = {}
        freq = {}
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all, batchsize=1)
        for index, batch in tqdm(enumerate(loader)):
            img, label, _, _, _ = batch
            with torch.no_grad():
                #x1, _ = self.model.forward(img.to(self.device))
                x1, _ = self.model.forward(img.to(self.device), source=False)
                pred = F.softmax(x1, dim=1)
            pred_probs = pred.max(dim=1)[0]
            pred_probs = pred_probs.squeeze()
            pred_label = torch.argmax(pred, dim=1).squeeze()
            for i in range(num_cls):
                cls_mask = pred_label == i
                cnt = cls_mask.sum()
                if cnt == 0:
                    continue
                cls_probs = torch.masked_select(pred_probs, cls_mask)
                cls_probs = cls_probs.detach().cpu().numpy().tolist()
                cls_probs.sort()
                if i not in probs:
                    probs[i] = cls_probs[::5] # reduce the consumption of memory
                else:
                    probs[i].extend(cls_probs[::5])

        growth = {}
        thres = {}
        for k in probs.keys():
            cls_prob = probs[k]
            cls_total = len(cls_prob)
            freq[k] = cls_total
            cls_prob = np.array(cls_prob)
            cls_prob = np.sort(cls_prob)
            index = int(cls_total * prop)
            cls_thres = cls_prob[-index]
            cls_thres2 = cls_prob[index]
            #if cls_thres == 1.0:
            #    cls_thres = 0.999
            thres[k] = cls_thres
        if self.config.source=='synthia':
            thres[9] = 1
            thres[14] = 1
            thres[16] = 1
        #for i in range(self.config.num_classes):
        #    if i in thres:
        #        continue
        #    else:
        #        thres[i] = 1
        print(thres)
        return thres

    def save_pred(self, round):
        # Using the threshold to generate pseudo labels and save
        print("[Generate pseudo labels]")
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all)
        interp = nn.Upsample(size=(1024, 2048), mode="bilinear", align_corners=True)

        self.plabel_path = osp.join(self.config.plabel, self.config.note, str(round))

        mkdir(self.plabel_path)
        self.config.target_data_dir = self.plabel_path
        self.pool = Pool() # save the probability of pseudo labels for the pixel-wise similarity matchinng, which is detailed around Eq. (9)
        accs = AverageMeter()  # Counter
        props = AverageMeter() # Counter
        cls_acc = GroupAverageMeter() # Class-wise Acc/Prop of Pseudo labels

        self.mean_memo = {i: [] for i in range(self.config.num_classes)}
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
                image, label, _, _, name = batch
                label = label.to(self.device)
                img_name = name[0].split("/")[-1]
                dir_name = name[0].split("/")[0]
                img_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                temp_dir = osp.join(self.plabel_path, dir_name)
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                #output, _ = self.model.forward(image.to(self.device))
                output, _ = self.model.forward(image.to(self.device), source=False)
                output = interp(output)
                # pseudo labels selected by glocal threshold
                mask, plabel = thres_cb_plabel(output, self.cb_thres, num_cls=self.config.num_classes)
                # pseudo labels selected by local threshold
                mask2, plabel2 = gene_plabel_prop( output, self.config.cb_prop)
                # mask fusion
                # The fusion strategy is detailed in Sec. 3.3 of paper
                mask, plabel = mask_fusion(output, mask, mask2)
                self.pool.update_pool(output, mask=mask.float())
                acc, prop, cls_dict = Acc(plabel, label, num_cls=self.config.num_classes)
                cnt = (plabel != 255).sum().item()
                accs.update(acc, cnt)
                props.update(prop, 1)
                cls_acc.update(cls_dict)
                plabel = plabel.view(1024, 2048)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                plabel = Image.fromarray(plabel)

                plabel.save("%s/%s.png" % (temp_dir, img_name.split(".")[0]))

        print('The Accuracy :{:.2%} and proportion :{:.2%} of Pseudo Labels'.format(accs.avg.item(), props.avg.item()))
        if self.config.neptune:
            neptune.send_metric("Acc", accs.avg)
            neptune.send_metric("Prop", props.avg)

    def save_model(self, iter):
        #tmp_name = "_".join(("GTA5", str(iter))) + ".pth"
        tmp_name = "_".join(("Synthia", str(iter))) + ".pth"
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], tmp_name))

    def validate(self):
        self.model = self.model.eval()
        testloader = dataset.init_test_dataset(self.config, self.config.target, set='val')
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        union = torch.zeros(self.config.num_classes, 1,dtype=torch.float).cuda().float()
        inter = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        preds = torch.zeros(self.config.num_classes, 1, dtype=torch.float).cuda().float()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(testloader)):
                image, label, _, _, name = batch
                #output, _ =  self.model(image.cuda())
                output, _ =  self.model(image.cuda(), source=False)
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
                class16_iou = torch.cat((iou[:9], iou[10:14], iou[15:16], iou[17:]))
                class16_miou = class16_iou.mean().item()
                class13_iou = torch.cat((class16_iou[:3], class16_iou[6:]))
                class13_miou = class13_iou.mean().item()
                print('16-Class mIoU:{:.2%}'.format(class16_miou))
                print('13-Class mIoU:{:.2%}'.format(class13_miou))
            #mIoU = iou.mean().item()
            #mAcc = acc.mean().item()
            #iou = iou.cpu().numpy()
            #print('mIoU: {:.2%} mAcc : {:.2%} '.format(mIoU, mAcc))
            if self.config.neptune:
                neptune.send_metric('mIoU', mIoU)
                neptune.send_metric('mAcc', mAcc)
        if self.config.source=='synthia':
            return class13_miou
        else:
            return iou.mean().item()

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
