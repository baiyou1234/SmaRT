from unet3d import UNet3d, TAdaBN3D
from autoaugment import LearnableImageNetPolicy
import torch
import os
from metrics import *
from augmodel import Augmentmodel
import numpy as np
from utils_brats_all import get_data_loader, parse_config, set_random
import monai.losses as losses
dice_loss = losses.DiceLoss()
from loss import CombinedLoss
from shape_analysis_functions import cnh_loss,ih_loss
import torch.nn.functional as F
import argparse


def test(config, upl_model, test_loader,exp_name,device, epoch = 0):
    num_classes = config['train']['num_classes']
    for data_loader in [test_loader]:  # 只对测试集进行评估
        all_batch_dice = []
        all_batch_assd = []
        all_batch_hd95 = []
        hd95_list_wt = []
        hd95_list_co = []
        hd95_list_ec = []  # 新增 HD95 列表
        IoU_list_wt = []
        IoU_list_co = []
        IoU_list_ec = []
        RVE_list_wt = []
        RVE_list_co = []
        RVE_list_ec = []
        sen_list_wt = []
        sen_list_co = []
        sen_list_ec = []
        output_result = []
        dice1_val = 0.0
        dice2_val = 0.0
        dice3_val = 0.0
        show = 0
        with torch.no_grad():
            upl_model.eval()
            skip_num = 0
            val_loss = 0
            dice1_val = 0
            dice2_val = 0
            dice3_val = 0
            RVE1_val = 0
            RVE2_val = 0
            RVE3_val = 0
            batch_count = 0
            num = 0
            for it, (image, label, xt_name, lab_Imag) in enumerate(test_loader):
                image = image.to(device, non_blocking=True)
                label = label.long().to(device, non_blocking=True).squeeze(1) 
                aux_seg_1 = upl_model.forward(image)
                dice1, dice2, dice3 = cal_dice(aux_seg_1, label)  
                hd95_ec, hd95_co, hd95_wt = cal_hd95(aux_seg_1, label)
                IoU_ec, IoU_co, IoU_wt = IoU(aux_seg_1, label)
                sen_ec, sen_co, sen_wt = cal_sensitivity(aux_seg_1, label)
                dice1_val += dice1
                dice2_val += dice2
                dice3_val += dice3
                batch_count += 1
                hd95mean = (hd95_wt + hd95_co + hd95_ec) / 3
                IoUmean = (IoU_wt + IoU_co + IoU_ec) / 3
                senmean = (sen_wt + sen_co + sen_ec) / 3
                dicemean = (dice1 + dice2 + dice3) / 3
                hd95_list_wt.append(hd95_wt)
                hd95_list_co.append(hd95_co)
                hd95_list_ec.append(hd95_ec)
                IoU_list_wt.append(IoU_wt)
                IoU_list_co.append(IoU_co)
                IoU_list_ec.append(IoU_ec)
                sen_list_wt.append(sen_wt)
                sen_list_co.append(sen_co)
                sen_list_ec.append(sen_ec)
                print(f"dice:[{dicemean}] hd95:[{hd95mean}] IoU:[{IoUmean}] sen:[{senmean}]")

        avg_dice1_val = dice1_val / (len(test_loader)-skip_num)
        avg_dice2_val = dice2_val / (len(test_loader)-skip_num)
        avg_dice3_val = dice3_val / (len(test_loader)-skip_num)
        avg_hd95_wt = np.nanmean(hd95_list_wt)
        avg_hd95_co = np.nanmean(hd95_list_co)
        avg_hd95_ec = np.nanmean(hd95_list_ec)
        avg_IoU_wt = np.nanmean(IoU_list_wt)
        avg_IoU_co = np.nanmean(IoU_list_co)
        avg_IoU_ec = np.nanmean(IoU_list_ec)
        avg_sen_wt = np.nanmean(sen_list_wt)
        avg_sen_co = np.nanmean(sen_list_co)
        avg_sen_ec = np.nanmean(sen_list_ec)
        dicemean = (avg_dice1_val + avg_dice2_val + avg_dice3_val) / 3
        output_result.append(f"ET : {avg_dice1_val}")
        output_result.append(f"TC : {avg_dice2_val}")
        output_result.append(f"WT : {avg_dice3_val}")
        output_result.append(f"HD95_ET : {avg_hd95_ec}")
        output_result.append(f"HD95_TC : {avg_hd95_co}")
        output_result.append(f"HD95_WT : {avg_hd95_wt}")
        output_result.append(f"IoU_ET : {avg_IoU_ec}")
        output_result.append(f"IoU_TC : {avg_IoU_co}")
        output_result.append(f"IoU_WT : {avg_IoU_wt}")
        output_result.append(f"sen_ET : {avg_sen_ec}")
        output_result.append(f"sen_TC : {avg_sen_co}")
        output_result.append(f"sen_WT : {avg_sen_wt}")
            
        print(skip_num)
        results_dir = f"results_comparison/tmi"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'new_origin_child_1.txt'), 'a') as file:
                for line in output_result:
                    file.write(line + "\n")
        return dicemean
                
def get_model_module(model):
    """
    动态获取模型的访问方式，支持单 GPU 和多 GPU。
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def momentum_update_key_encoder(model, momentum_model):
    """
    Momentum update of the key encoder
    """
    # encoder_q -> encoder_k
    for param_q, param_k in zip(
        model.parameters(), momentum_model.parameters()
    ):
        param_k.data = param_k.data * 0.95 + param_q.data * 0.05
    return momentum_model

def total_variation_loss_3d(pred):
    """
    pred: [B, C, D, H, W] - softmax输出
    惩罚预测体积中相邻体素间的跳变
    """
    dz = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dx = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    return (dz.mean() + dy.mean() + dx.mean())

def gradient_loss_3d(pred):
    """
    pred: [B, C, D, H, W] - softmax输出
    用三维一阶差分计算梯度
    """
    dz = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    return (dz.abs().mean() + dy.abs().mean() + dx.abs().mean())

def compactness_loss_3d(pred, epsilon=1e-6):
    """
    pred: [B, C, D, H, W] - softmax输出
    近似方式：用TV估算表面积，面积用预测体素求和
    """
    volume = torch.sum(pred, dim=[2, 3, 4]) + epsilon  # [B, C]
    surface = total_variation_loss_3d(pred) * pred.shape[2] * pred.shape[3] * pred.shape[4]
    loss = torch.mean(surface / volume)
    return loss
    
def train(config,train_loader,test_loader,source_model):
    print("train")
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    device = config['train']['gpu']
    checkpoint = torch.load(source_model)
    upl_model = UNet3d().to(device)
    upl_model.load_state_dict(checkpoint)
    momentum_model = UNet3d().to(device)
    momentum_model.load_state_dict(checkpoint)
    print('source_model_created')
    class_weights = torch.tensor([1.0, 2.0, 1.5, 1.5], device=device)
    dice_reduction='macro'
    criterion = CombinedLoss(
        ce_weight=3.0,
        dice_weight=2.0,
        dice_reduction=dice_reduction,
        class_weights=class_weights,
        device=device
    )
    aug_num = 1
    print('source_model_loaded')
    aug_model = Augmentmodel(upl_model).to(device)
    aug_momentum_model = Augmentmodel(momentum_model).to(device)
    #test(config,upl_model,test_loader,exp_name=exp_name,device = device,epoch = 0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, aug_model.parameters()),lr=config['train']['lr'])
    num_epochs = 1000
    best_dice = 0.
    output_dir = "validation_results_ssa-t2f"
    os.makedirs(output_dir,exist_ok=True)
    train_flag = True
    aug_model = aug_model.to(device)
    for epoch in range(num_epochs):
        #print(epoch)
        if train_flag:
            pesudo_labels = []
            with torch.no_grad():
                for i, (B, B_label, _, _) in enumerate(train_loader):
                    B = B.to(device)
                    pesudo_label = aug_momentum_model(B, aug = 0)
                    pesudo_labels.append(pesudo_label.detach())
            aug_model.train()
            for module in aug_model.modules():
                if isinstance(module, TAdaBN3D) or isinstance(module, LearnableImageNetPolicy):
                    module.train()
                else:
                    module.eval()
            for i, (B, B_label, _, name) in enumerate(train_loader):
                B_label = B_label.long().squeeze(1).to(device)
                B = B.to(device)
                optimizer.zero_grad() 
                for j in range(aug_num):
                    out, weight = aug_model(B, aug = 0)
                    weight = weight.to(device)
                    total_loss = torch.tensor(0.0).to(device)
                    
                    out = aug_model(B, aug = 0)
                   
                    for k in range(out.size(0)):
                        input_i = out[k].unsqueeze(0)
                        loss_dice = criterion(input_i, B_label.unsqueeze(1)) 
                        loss_ao = bh_loss(input_i) / 10
                        loss_tu = f_tu(input_i)
                        weighted_loss_i = (loss_ao + loss_tu + loss_dice) * weight[k]
                        print(loss_dice,loss_ao,loss_tu)
                        total_loss += weighted_loss_i
                    if total_loss != 0:
                        total_loss.backward() 
                optimizer.step()
                
        # # valid for target domain
        if (epoch+1) % 1 == 0:
            current_dice = test(config,upl_model,test_loader,exp_name=exp_name,device = device,epoch = epoch)
            if (current_dice) > best_dice:
                best_dice = current_dice
                model_dir = "/data/birth/cyf/output/wyh_output/tta/new_origin_5/" + str(exp_name )
                os.makedirs(model_dir, exist_ok=True)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                torch.save(upl_model.state_dict(), best_epoch)
            model_dir = "/data/birth/cyf/output/wyh_output/tta/new_origin_5/" + str(exp_name )
            os.makedirs(model_dir, exist_ok=True)
            best_epoch = '{}/model-{}.pth'.format(model_dir, str(epoch))
            torch.save(upl_model.state_dict(), best_epoch)   
        momentum_model = momentum_update_key_encoder(upl_model,momentum_model)
    if train_flag and (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))

    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=True)
    upl_model.eval()
    print("test")
    test(config,upl_model,test_loader,exp_name=exp_name,device = device)
    
def main():
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train3d.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    source_model = '/home/cyf/TTA/unet3d_best.pth'
    batch_train = 1
    batch_test = 1
    num_workers = 0
    source_root = '/FM_data/cyf/tta_data/BraTS2024'
    target_root = '/data/birth/cyf/shared_data/TTA/tta_data/BRATS-PED'
    train_path = 'train'
    test_path = 'test'
    mode = 'target_to_target'
    img = 'all'
    train_loader,test_loader = get_data_loader(source_root,target_root,
                                               train_path,test_path,
                                               batch_train,batch_test,
                                               nw = num_workers,
                                               img=img,mode=mode)
    print("数据加载完成")

    train(config,train_loader,test_loader,source_model)
        
if __name__ == '__main__':
    
    set_random()
    torch.manual_seed(0.95)
    torch.cuda.manual_seed(0.95) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    main()