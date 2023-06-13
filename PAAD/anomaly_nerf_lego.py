import os
import random
from copy import deepcopy

import cv2
import imageio
import numpy as np
import skimage
import torch
from torch.utils.data import DataLoader
import matplotlib as plt

import datasets.LEGO_3D as lego
from datasets.LEGO_3D import LEGODataset
from inerf_helpers import camera_transf
from sklearn.metrics import roc_auc_score
from nerf_helpers import load_nerf
from render_helpers import get_rays, render, to8b
from utils import (MAPE, Relative_L2, calculate_resmaps, config_parser, find_nearest, find_POI,
                   img2mse, load_blender, load_blender_AD, load_blender_ad,
                   load_llff_data, pose_retrieval, pose_retrieval_efficient, pose_retrieval_loftr, resmaps_ssim, show_img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1024
random.seed(seed)
torch.manual_seed(seed)


def run():

    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    lrate = args.lrate
    sampling_strategy = args.sampling_strategy

    class_names = lego.CLASS_NAMES if args.class_name == 'all' else [
        args.class_name]

    for class_name in class_names:
        # load the good imgs with their poses
        imgs, hwf, poses = load_blender_ad(
            args.data_dir, model_name, args.half_res, args.white_bkgd)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender

        # load the anomaly image
        lego_dataset = LEGODataset(dataset_path=args.data_dir,
                                   class_name=class_name,
                                   resize=400)

        lego_loader = DataLoader(dataset=lego_dataset,
                                 batch_size=1,
                                 pin_memory=False)
        test_imgs = list()
        gt_mask_list = list()
        gt_list = list()
        score_map_list = list()
        pred_list = list()
        index = 0
        for x, y, mask in lego_loader:
            test_imgs.extend(x.cpu().numpy())
            gt_list.extend(y.cpu().numpy())
            mask = (mask.cpu().numpy()/255.0).astype(np.uint8)
            gt_mask_list.extend(mask)

            obs_img = x.cpu().numpy().squeeze(axis=0)
            # Find the start pose by looking for the most similar images
            # start_pose=find_nearest(imgs,obs_img,poses,method='lpips')
            # start_pose = pose_retrieval(imgs, obs_img, poses)
            start_pose = pose_retrieval_loftr(imgs, obs_img, poses)
            # find points of interest of the observed image
            # xy pixel coordinates of points of interest (N x 2)
            POI = find_POI(obs_img, DEBUG)
            obs_img = (np.array(obs_img) / 255.).astype(np.float32)

            # create meshgrid from the observed image
            coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                                dtype=int)

            # create sampling mask for interest region sampling strategy
            interest_regions = np.zeros((H, W, ), dtype=np.uint8)
            interest_regions[POI[:, 1], POI[:, 0]] = 1
            I = args.dil_iter
            interest_regions = cv2.dilate(interest_regions, np.ones(
                (kernel_size, kernel_size), np.uint8), iterations=I)
            interest_regions = np.array(interest_regions, dtype=bool)
            interest_regions = coords[interest_regions]

            # not_POI -> contains all points except of POI
            coords = coords.reshape(H * W, 2)
            not_POI = set(tuple(point) for point in coords) - \
                set(tuple(point) for point in POI)
            not_POI = np.array([list(point) for point in not_POI]).astype(int)

            # Load NeRF Model
            render_kwargs = load_nerf(args, device)
            bds_dict = {
                'near': near,
                'far': far,
            }
            render_kwargs.update(bds_dict)

            # Create pose transformation model
            start_pose = torch.Tensor(start_pose).to(device)
            cam_transf = camera_transf().to(device)
            optimizer = torch.optim.Adam(
                params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))
            testsavedir = os.path.join(
                output_dir, model_name, "loftr",str(model_name)+"_"+str(index))
            os.makedirs(testsavedir, exist_ok=True)

            # imgs - array with images are used to create a video of optimization process
            if OVERLAY is True:
                gif_imgs = []
            for k in range(300):

                if sampling_strategy == 'random':
                    rand_inds = np.random.choice(
                        coords.shape[0], size=batch_size, replace=False)
                    batch = coords[rand_inds]

                elif sampling_strategy == 'interest_points':
                    if POI.shape[0] >= batch_size:
                        rand_inds = np.random.choice(
                            POI.shape[0], size=batch_size, replace=False)
                        batch = POI[rand_inds]
                    else:
                        batch = np.zeros((batch_size, 2), dtype=np.int)
                        batch[:POI.shape[0]] = POI
                        rand_inds = np.random.choice(
                            not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                        batch[POI.shape[0]:] = not_POI[rand_inds]

                elif sampling_strategy == 'interest_regions':
                    rand_inds = np.random.choice(
                        interest_regions.shape[0], size=batch_size, replace=False)
                    batch = interest_regions[rand_inds]

                else:
                    print('Unknown sampling strategy')
                    return

                target_s = obs_img[batch[:, 1], batch[:, 0]]
                target_s = torch.Tensor(target_s).to(device)
                pose = cam_transf(start_pose)

                rays_o, rays_d = get_rays(
                    H, W, focal, pose)  # (H, W, 3), (H, W, 3)
                rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
                rays_d = rays_d[batch[:, 1], batch[:, 0]]
                batch_rays = torch.stack([rays_o, rays_d], 0)

                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=k < 10, retraw=True,
                                                **render_kwargs)

                optimizer.zero_grad()
                loss = img2mse(rgb, target_s)
                # loss = Relative_L2(rgb,target_s)
                # loss = MAPE(rgb,target_s)
                loss.backward()
                optimizer.step()

                new_lrate = lrate * (0.8 ** ((k + 1) / 100))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                if (k + 1) % 50 == 0 or k == 0:
                    print('Step: ', k)
                    print('Loss: ', loss)

                    if OVERLAY is True:
                        with torch.no_grad():
                            rgb, disp, acc, _ = render(
                                H, W, focal, chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                            rgb = rgb.cpu().detach().numpy().astype(np.float32)
                            rgb8 = to8b(rgb)
                            ref = to8b(obs_img)
                            filename = os.path.join(testsavedir, str(k)+'.png')
                            dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                            imageio.imwrite(filename, dst)
                            gif_imgs.append(dst)
            # quality = 8 for mp4 format
            imageio.mimwrite(os.path.join(
                testsavedir, 'video.gif'), gif_imgs, fps=8)
            imageio.imwrite(os.path.join(testsavedir, 'ref.png'), ref)
            imageio.imwrite(os.path.join(testsavedir, 'rgb8.png'), rgb8)
            # s, resmap = calculate_resmaps(ref, rgb8, "ssim")
            # resmap = np.asarray(resmap*255, dtype=np.uint8)
            # dst = 255-resmap
            # threshold = 150
            # retval, pred_mask = cv2.threshold(dst, threshold, 255, 0)
            # imageio.imwrite(os.path.join(testsavedir, 'pred.png'), pred_mask)
            index = index+1
            # pred_mask = (np.array(pred_mask)/255.0).astype(np.uint8)
            # score_map_list.extend(dst)
            # if np.all(pred_mask == 0):
            #     pred_list.extend([0])
            # else:
            #     pred_list.extend([1])
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # fig_img_rocauc = ax[0]
        # fig_pixel_rocauc = ax[1]

        # r'Image-level AUROC'
        # per_image_rocauc = roc_auc_score(
        #     np.array(gt_list), np.array(pred_list))
        # print('image ROCAUC: %.3f ' % (per_image_rocauc))

        # r'Pixel-level AUROC'
        # flatten_gt_mask_list = np.concatenate(
        #     gt_mask_list).ravel().astype(np.uint8)
        # flatten_score_map_list = np.concatenate(
        #     score_map_list).ravel().astype(np.uint8)
        # per_pixel_rocauc = roc_auc_score(
        #     flatten_gt_mask_list, flatten_score_map_list)
        # print('pixel ROCAUC: %.3f ' % (per_pixel_rocauc))


DEBUG = False
OVERLAY = True

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
