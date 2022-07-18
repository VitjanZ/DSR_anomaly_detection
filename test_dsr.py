import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule, UpsamplingModule
from discrete_model import DiscreteLatentModel
import sys
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader_test import TestMVTecDataset
import cv2


def crop_image(image, img_dim):
    b,c,h,w = image.shape
    hdif = max(0,h - img_dim) // 2
    wdif = max(0,w - img_dim) // 2
    image_cropped = image[:,:,hdif:-hdif,wdif:-wdif]
    return image_cropped

def evaluate_model(model, model_normal, model_normal_top, model_decode, decoder_seg, model_upsample, obj_name, mvtec_path, cnt_total):
    img_dim = 256
    dataset = TestMVTecDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim,img_dim])

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)


    img_dim = 224
    total_pixel_scores = np.zeros((img_dim * img_dim * 500))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * 500))
    mask_cnt = 0

    total_gt = []
    total_score = []
    iter = cnt_total

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0,0]
        total_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask = crop_image(true_mask, img_dim)
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        loss_b, loss_t, data_recon, embeddings_t, embeddings = model(gray_batch)
        embeddings = embeddings.detach()
        embeddings_t = embeddings_t.detach()

        embedder = model._vq_vae_bot
        embedder_top = model._vq_vae_top

        anomaly_embedding_copy = embeddings.clone()
        anomaly_embedding_top_copy = embeddings_t.clone()
        recon_feat, recon_embeddings, _ = model_normal(anomaly_embedding_copy, embedder)
        recon_feat_top, recon_embeddings_top, loss_b_top = model_normal_top(anomaly_embedding_top_copy,
                                                                            embedder_top)

        up_quantized_recon_t = model.upsample_t(recon_embeddings_top)
        quant_join = torch.cat((up_quantized_recon_t, recon_embeddings), dim=1)
        recon_image_recon = model_decode(quant_join)

        up_quantized_embedding_t = model.upsample_t(embeddings_t)
        quant_join_real = torch.cat((up_quantized_embedding_t, embeddings), dim=1)
        recon_image = model._decoder_b(quant_join_real)
        out_mask = decoder_seg(recon_image_recon.detach(),
                               recon_image.detach())
        out_mask_sm = torch.softmax(out_mask, dim=1)

        upsampled_mask = model_upsample(recon_image_recon.detach(), recon_image.detach(), out_mask_sm)
        out_mask_sm_up = torch.softmax(upsampled_mask, dim=1)
        out_mask_sm_up = crop_image(out_mask_sm_up, img_dim)

        iter += 1


        out_mask_cv = out_mask_sm_up[0,1,:,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:,1:,:,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        total_score.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    auroc = roc_auc_score(total_gt, total_score)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    ap = average_precision_score(total_gt, total_score)
    print(obj_name+" AUC Image: "+str(auroc)+",  AUC Pixel: "+str(auroc_pixel)+", AP Pixel:"+str(ap_pixel)+", AP :"+str(ap))

    return auroc, auroc_pixel, ap_pixel, ap, iter

def train_on_device(obj_names, mvtec_path, run_basename):
    auroc_list = []
    auroc_pixel_list = []
    ap_pixel_list = []
    ap_list = []
    cnt_total = 0
    for obj_name in obj_names:
        run_name_pre = 'vq_model_pretrained_128_4096'

        run_name = run_basename+'_'

        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        embedding_dim = 128
        num_embeddings = 4096
        commitment_cost = 0.25
        decay = 0.99
        model_vq = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                      num_embeddings, embedding_dim,
                      commitment_cost, decay)
        model_vq.cuda()
        model_vq.load_state_dict(
            torch.load("./checkpoints/" + run_name_pre + ".pckl", map_location='cuda:0'))
        model_vq.eval()



        sub_res_hi_module = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_hi_module.load_state_dict(
            torch.load("./checkpoints/" + run_name + "subspace_restriction_hi_"+obj_name+".pckl", map_location='cuda:0'))
        sub_res_hi_module.cuda()
        sub_res_hi_module.eval()

        sub_res_lo_module = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_lo_module.load_state_dict(
            torch.load("./checkpoints/" + run_name + "subspace_restriction_lo_"+obj_name+".pckl", map_location='cuda:0'))
        sub_res_lo_module.cuda()
        sub_res_lo_module.eval()


        anom_det_module = AnomalyDetectionModule(embedding_size=embedding_dim)
        anom_det_module.load_state_dict(
            torch.load("./checkpoints/" + run_name + "anomaly_det_module_"+obj_name+".pckl", map_location='cuda:0'))
        anom_det_module.cuda()
        anom_det_module.eval()

        upsample_module = UpsamplingModule(embedding_size=embedding_dim)
        upsample_module.load_state_dict(
            torch.load("./checkpoints/" + run_name + "upsample_module_"+obj_name+".pckl", map_location='cuda:0'))
        upsample_module.cuda()
        upsample_module.eval()


        image_recon_module = ImageReconstructionNetwork(embedding_dim * 2,
                   num_hiddens,
                   num_residual_layers,
                   num_residual_hiddens)
        image_recon_module.load_state_dict(
            torch.load("./checkpoints/" + run_name + "image_recon_module_"+obj_name+".pckl", map_location='cuda:0'), strict=False)
        image_recon_module.cuda()
        image_recon_module.eval()


        with torch.no_grad():
            auroc, auroc_pixel, ap_pixel, ap, cnt = evaluate_model(model_vq, sub_res_hi_module, sub_res_lo_module, image_recon_module, anom_det_module, upsample_module, obj_name, mvtec_path, cnt_total)
            cnt_total += cnt
            ap_list.append(ap)
            auroc_list.append(auroc)
            auroc_pixel_list.append(auroc_pixel)
            ap_pixel_list.append(ap_pixel)

    print(run_basename)
    auroc_mean = np.mean(auroc_list)
    auroc_pixel_mean = np.mean(auroc_pixel_list)
    print("Detection AUROC: "+str(auroc_mean))
    print("Localization AUROC: "+str(auroc_pixel_mean))
    print("Localization AP: "+str(np.mean(ap_pixel_list)))


if __name__=="__main__":
    obj_names = ['capsule', 'bottle', 'grid', 'leather', 'pill', 'tile', 'transistor', 'zipper', 'cable', 'carpet',
                 'hazelnut', 'metal_nut', 'screw', 'toothbrush', 'wood']

    list1 = sys.argv[1].split(',')

    with torch.cuda.device(int(sys.argv[1])):
        train_on_device(obj_names, sys.argv[2], sys.argv[3])

