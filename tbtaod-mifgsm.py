"""
    Attack object detectors in a blackbox setting
    design blackbox loss
"""
# https://github.com/open-mmlab/mmcv#installation
import os
import sys
from pathlib import Path
from collections import defaultdict
import json as JSON
import random
import pdb
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
import datetime
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import argparse


mmdet_root = Path('mmdetection/')
sys.path.insert(0, str(mmdet_root))
from utils_mmdet import vis_bbox, VOC_BBOX_LABEL_NAMES, COCO_BBOX_LABEL_NAMES, voc2coco, get_det, is_success, get_iou
from utils_mmdet import model_train


# 加载Stable Diffusion模型
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_id = "/home/wcq/CompVis-stable-diffusion-v1-4"
sdm = StableDiffusionImg2ImgPipeline.from_pretrained("/home/wcq").to(device)

os.makedirs("/home/wcq/EBAD-main/imageshow/", exist_ok=True)


# 使用扩散模型生成合成图片
def generate_images(original_image, num_images=3, prompt="A picture of a similar style", strength=0.6):
    generated_images = []
    target_height, target_width = original_image.shape[2], original_image.shape[3]

    for i in range(num_images):
        generated_image = sdm(prompt=prompt, image=original_image, strength=strength, disable_tqdm=True).images[0]
        generated_image = generated_image.resize((target_width,target_height))

        # generated_image 是一个 PIL 图像
        generated_image = torch.from_numpy(np.array(generated_image)).permute(2, 0, 1).unsqueeze(0)

        # 移动到 GPU
        generated_image = generated_image
        generated_images.append(generated_image.cuda())

    return generated_images

# 将对抗样本与生成图像进行混合
def blend_image(original_image, generated_images, eta):
    blended_images = []
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for j, img in enumerate(generated_images):
        # 融合
        blended_image = (1 - eta) * original_image + eta * img

        blended_images.append(blended_image)
        
    return blended_images


def PM_tensor_weight_balancing(im, adv, target, w, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False):
    """perturbation machine, balance the weights of different surrogate models
    args:
        im (tensor): original image, shape [1,3,h,w].cuda()
        adv (tensor): adversarial image
        target (numpy.ndarray): label for object detection, (xyxy, cls, conf)
        w (numpy.ndarray): ensemble weights
        ensemble (): surrogate ensemble
        eps (int): linf norm bound (0-255)
        n_iters (int): number of iterations
        alpha (flaot): step size

    returns:
        adv_list (list of Tensors): list of adversarial images for all iterations
        LOSS (dict of lists): 'ens' is the ensemble loss, and other individual surrogate losses
    """
    # prepare target label input: voc -> coco, since models are trained on coco
    bboxes_tgt = target[:,:4].astype(np.float32)
    labels_tgt = target[:,4].astype(int).copy()
    if dataset == 'voc':
        for i in range(len(labels_tgt)): 
            labels_tgt[i] = voc2coco[labels_tgt[i]]

    generated_images=generate_images(im)
    
    im_np = im.squeeze().cpu().numpy().transpose(1, 2, 0)
    adv_list = []
    pert = adv - im
    LOSS = defaultdict(list) # loss lists for different models
    momentum = torch.zeros_like(pert)
    beta = ((eps*2)/255)*2
    for i in range(20):
        pert.requires_grad = True
        blended_images=blend_image(adv,generated_images,0.6)
        loss_list = []
        loss_list_np = []
        for model in ensemble:
            # 原始对抗样本的损失
            loss1 = model.loss(im_np, pert, bboxes_tgt, labels_tgt)
            loss_list.append(loss1)
            loss_list_np.append(loss1.item())
            LOSS[model.model_name].append(loss1.item())

            loss2 = torch.zeros_like(loss1)
            # 遍历所有生成图片
            total_grad = None  # 初始化总梯度

            for _, img in enumerate(blended_images):
                for i in range(5):
                    # 创建邻近图像，通过添加随机噪声实现
                    img_neighbor = img + torch.rand_like(img) * (2 * beta) - beta
                    pert1 = img_neighbor - im  # 计算扰动

                    pert1.requires_grad = True
                    # 计算损失并进行反向传播
                    loss2 = model.loss(im_np, pert1, bboxes_tgt, labels_tgt)
                    loss2.backward()

                     # 获取当前梯度
                    if total_grad is None:
                        total_grad = pert1.grad.clone()  # 初始化总梯度
                    else:
                        total_grad += pert1.grad  # 累加梯度

                    # 清除梯度以便下一次迭代
                    pert1.grad = None  # 防止梯度累积
            sample_grad = total_grad / (15)

        if weight_balancing:
            w_inv = 1/np.array(loss_list_np)
            w = w_inv / w_inv.sum()

        # print(f"w: {w}")
        loss_ens = sum(w[i]*loss_list[i] for i in range(len(ensemble)))
        loss_ens.backward()
        with torch.no_grad():
            pert_grad=pert.grad
            # 展平张量
            pert_grad_flat = pert_grad.reshape(-1)
            sample_grad_flat = sample_grad.reshape(-1)

            # 计算余弦相似性
            cossim = torch.nn.functional.cosine_similarity(pert_grad_flat, sample_grad_flat, dim=0)
            for _ in range(len(loss1.shape) - 1):
                cossim = cossim.unsqueeze(-1)
            final_grad = cossim*pert_grad + (1-cossim)*sample_grad
            momentum = 0.9 * momentum + final_grad / final_grad.abs().max()
            pert = pert - alpha*torch.sign(momentum)
            LOSS['ens'].append(loss_ens.item())
            adv = (im + pert).clip(0, 255)
            adv_list.append(adv)
    return adv_list, LOSS


def PM_tensor_weight_balancing_np(im_np, target, w_np, ensemble, eps, n_iters, alpha, dataset='voc', weight_balancing=False, adv_init=None):
    """perturbation machine, numpy input version
    
    """
    device = next(ensemble[0].parameters()).device
    im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).permute(2,0,1).unsqueeze(0).float().to(device)

    adv_list, LOSS = PM_tensor_weight_balancing(im, adv, target, w_np, ensemble, eps, n_iters, alpha, dataset, weight_balancing)
    adv_np = adv_list[-1].squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return adv_np, LOSS


def get_bb_loss(detections, target_clean, LOSS):
    """define the blackbox attack loss
        if the original object is detected, the loss is the conf score of the victim object
        otherwise, the original object disappears, the conf is below the threshold, the loss is the wb ensemble loss
    args:
        detections ():
        target_clean ():
        LOSS ():
    return:
        bb_loss (): the blackbox loss
    """
    max_iou = 0
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > max(max_iou, 0.3) and items[4] == target_clean[0][4]:
            max_iou = iou
            bb_loss = 1e3 + items[5] # add a large const to make sure it is larger than conf ens loss

    # if it disappears
    if max_iou < 0.3:
        bb_loss = LOSS['ens'][-1]

    return bb_loss


def save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query):    
    """get the loss bb, success_list on all surrogate models, and save detections to fig
    
    args:

    returns:
        loss_bb (float): loss on the victim model
        success_list (list of 0/1s): successful for all models
    """
    fig_h = 5
    fig_w = 5
    n_all = len(all_models)
    fig, ax = plt.subplots(2,1+n_all,figsize=((1+n_all)*fig_w,2*fig_h))
    # 1st row, clean image, detection on surrogate models, detection on victim model
    # 2nd row, perturbed image, detection on surrogate models, detection on victim model
    row = 0
    ax[row,0].imshow(im_np)
    ax[row,0].set_title('clean image')
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(im_np)
        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(im_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

    row = 1
    ax[row,0].imshow(adv_np)
    ax[row,0].set_title(f'adv image @ iter {n_query} \n {attack_goal}')
    success_list = [] # 1 for success, 0 for fail for all models
    for model_idx, model in enumerate(all_models):
        det_adv = model.det(adv_np)
        bboxes, labels, scores = det_adv[:,:4], det_adv[:,4], det_adv[:,5]
        vis_bbox(adv_np, bboxes, labels, scores, ax=ax[row,model_idx+1], dataset=dataset)
        ax[row,model_idx+1].set_title(model.model_name)

        # check for success and get bb loss
        if model_idx == n_all-1:
            loss_bb = get_bb_loss(det_adv, target_clean, LOSS)

        # victim model is at the last index
        success_list.append(is_success(det_adv, target_clean))
    
    plt.tight_layout()
    if success_list[-1]:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}_success.png")
    else:
        plt.savefig(log_root / f"{im_idx}_{im_id}_iter{n_query}.png")
    plt.close()

    return loss_bb, success_list
    

def main():
    parser = argparse.ArgumentParser(description="generate perturbations")
    parser.add_argument("--eps", type=int, default=20, help="perturbation level: 10,20,30,40,50")
    parser.add_argument("--iters", type=int, default=20, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--root", type=str, default='result', help="the folder name of result")
    parser.add_argument("--victim", type=str, default='FreeAnchor', help="victim model")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--surrogate", type=str, default='Faster R-CNN', help="surrogate model")
    parser.add_argument("--dataset", type=str, default='voc', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    args = parser.parse_args()
    
    eps = args.eps
    n_iters = args.iters
    x_alpha = args.x
    alpha = eps / n_iters * x_alpha
    dataset = args.dataset
    victim_name = args.victim

    # load surrogate models
    ensemble = []
    models_all = ['Faster R-CNN', 'YOLOv3', 'FCOS', 'Grid R-CNN', 'SSD']
    model_list = models_all[:1]
    model_list = [args.surrogate]
    for model_name in model_list:
        ensemble.append(model_train(model_name=model_name, dataset=dataset))

    # load victim model
    # ['RetinaNet', 'Libra', 'FoveaBox', 'FreeAnchor', 'DETR', 'Deformable']
    if victim_name == 'Libra':
        victim_name = 'Libra R-CNN'
    elif victim_name == 'Deformable':
        victim_name = 'Deformable DETR'

    model_victim = model_train(model_name=victim_name, dataset=dataset)
    all_model_names = model_list + [victim_name]
    all_models = ensemble + [model_victim]


    # create folders
    exp_name = f'BB_wb_linf_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}'
    if dataset != 'voc':
        exp_name += f'_{dataset}'
    
    exp_name += f'_{args.surrogate}'
    exp_name += '_single'

    print(f"\nExperiment: {exp_name} \n")
    result_root = Path(f"abution/time/results_detection_voc_dm5_sam20_new_try_20/")
    exp_root = result_root / exp_name
    log_root = exp_root / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log_loss_root = exp_root / 'logs_loss'
    log_loss_root.mkdir(parents=True, exist_ok=True)
    adv_root = exp_root / 'advs'
    adv_root.mkdir(parents=True, exist_ok=True)
    target_root = exp_root / 'targets'
    target_root.mkdir(parents=True, exist_ok=True)

    test_image_ids = JSON.load(open(f"data/{dataset}_2to6_objects.json"))
    data_root = Path("/home/wcq/EBAD-main/data/SalmanAsif")
    if dataset == 'voc':
        im_root = data_root / "VOC/VOC2007/JPEGImages/"
        n_labels = 20
        label_names = VOC_BBOX_LABEL_NAMES
    else:
        im_root = data_root / "COCO/val2017/"
        n_labels = 80
        label_names = COCO_BBOX_LABEL_NAMES

    dict_k_sucess_id_v_query = {} # successful im_ids
    dict_k_valid_id_v_success_list = {} # lists of success for all mdoels for valid im_ids
    n_obj_list = []

    for im_idx, im_id in tqdm(enumerate(test_image_ids[:10])):
        im_path = im_root / f"{im_id}.jpg"
        im_np = np.array(Image.open(im_path).convert('RGB'))
        
        # get detection on clean images and determine target class
        det = model_victim.det(im_np)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        print(f"n_objects: {len(det)}")
        n_obj_list.append(len(det))
        if len(det) == 0: # if nothing is detected, skip this image
            continue
        else:
            dict_k_valid_id_v_success_list[im_id] = []

        all_categories = set(labels.astype(int))  # all apperaing objects in the scene
        # randomly select a victim
        victim_idx = random.randint(0,len(det)-1)
        victim_class = int(det[victim_idx,4])

        # randomly select a target
        select_n = 1 # for each victim object, randomly select 5 target objects
        target_pool = list(set(range(n_labels)) - all_categories)
        target_pool = np.random.permutation(target_pool)[:select_n]

        # for target_class in target_pool:
        target_class = int(target_pool[0])

        # basic information of attack
        attack_goal = f"{label_names[victim_class]} to {label_names[target_class]}"
        info = f"im_idx: {im_idx}, im_id: {im_id}, victim_class: {label_names[victim_class]}, target_class: {label_names[target_class]}\n"
        print(info)
        file = open(exp_root / f'{exp_name}.txt', 'a')
        file.write(f"{info}\n\n")
        file.close()

        target = det.copy()
        # only change one label
        target[victim_idx, 4] = target_class
        # only keep one label
        target_clean = target[victim_idx,:][None]

        # target = target_clean

        # save target to np
        np.save(target_root/f"{im_id}_target", target)
        w_np = np.ones(1) / 1

        adv_np, LOSS = PM_tensor_weight_balancing_np(im_np, target, w_np, ensemble, eps, n_iters, alpha=alpha, dataset=dataset)
        n_query = 0
        loss_bb, success_list = save_det_to_fig(im_np, adv_np, LOSS, target_clean, all_models, im_id, im_idx, attack_goal, log_root, dataset, n_query)
        dict_k_valid_id_v_success_list[im_id].append(success_list)

        # save adv in folder
        adv_path = adv_root / f"{im_id}_iter{n_query:02d}.png"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)

        # successful
        if success_list[-1]:
            dict_k_sucess_id_v_query[im_id] = n_query
            print(f"success! image im idx: {im_idx}")
            
        w_list = []
        loss_bb_list = [loss_bb]
        loss_ens_list = LOSS['ens'] # ensemble losses during training
        

        if im_id in dict_k_sucess_id_v_query:
            # save to txt
            info = f"im_idx: {im_idx}, id: {im_id}, loss_bb: {loss_bb:.4f}, w: {w_np}\n"
            file = open(exp_root / f'{exp_name}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"im_idx: {im_idx}; total_success: {len(dict_k_sucess_id_v_query)}")

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_ens_list)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        im = im_np
        im_temp = im if model_victim.rgb else im[:,:,::-1]
        det = get_det(model_victim.model, victim_name, im_temp, dataset)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        vis_bbox(im, bboxes, labels, scores, ax=ax[1], dataset=dataset)
        ax[1].set_title(f"clean image")

        adv = adv_np
        im_temp = adv if model_victim.rgb else adv[:,:,::-1]
        det = get_det(model_victim.model, victim_name, im_temp, dataset)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        vis_bbox(adv, bboxes, labels, scores, ax=ax[2], dataset=dataset)
        ax[2].set_title(f'adv image @ iter {n_query} \n {label_names[victim_class]} to {label_names[target_class]}')
        ax[3].plot(loss_bb_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        ax[4].plot(w_list)
        ax[4].legend(model_list, shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        ax[4].set_yscale('log')
        plt.tight_layout()
        if im_id in dict_k_sucess_id_v_query:
            plt.savefig(log_loss_root / f"{im_id}_success_iter{n_query}.png")
        else:
            plt.savefig(log_loss_root / f"{im_id}.png")
        plt.close()

        if len(dict_k_sucess_id_v_query) > 0:
            print(f"success rate (victim): {len(dict_k_sucess_id_v_query) / len(dict_k_valid_id_v_success_list)}")

        # print surrogate success rates
        success_list_stack = []
        for valid_id in dict_k_valid_id_v_success_list:
            success_list = np.array(dict_k_valid_id_v_success_list[valid_id])
            success_list = success_list.sum(axis=0).astype(bool).astype(int).tolist()
            success_list_stack.append(success_list)

        success_list_stack = np.array(success_list_stack).sum(axis=0)
        # pdb.set_trace()
        for idx, success_cnt in enumerate(success_list_stack):
            print(f"success rate of {all_model_names[idx]}: {success_cnt / len(dict_k_valid_id_v_success_list)}")
    
        # save np files / save at each iteration in case got cut off in the middle
        np.save(exp_root/f"dict_k_sucess_id_v_query", dict_k_sucess_id_v_query)
        np.save(exp_root/f"dict_k_valid_id_v_success_list", dict_k_valid_id_v_success_list)


if __name__ == '__main__':
    main()
