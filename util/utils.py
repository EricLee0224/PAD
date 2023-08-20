from copy import deepcopy
import importlib
import json
import os
from operator import index

import cv2
import imageio
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from util.model_helper import ModelHelper
from retrieval.loftr import LoFTR, default_cfg
from retrieval.retrieval import *
from easydict import EasyDict
import yaml


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='./data/nerf_synthetic/',
                        help='path to folder with synthetic or llff data')
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--model_name", type=str,
                        help='name of the nerf model')
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help='where to store output images/videos')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts',
                        help='folder with saved checkpoints')
    parser.add_argument("--ckpt_name", type=str, 
                        help='name of ckpt')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--chunk", type=int, default=1024*32,  # 1024*32
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,  # 1024*64
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=0.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')

    # blender options
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff options
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # iNeRF options
    parser.add_argument("--obs_img_num", type=int, default=0,
                        help='Number of an observed image')
    parser.add_argument("--dil_iter", type=int, default=1,
                        help='Number of iterations of dilation process')
    parser.add_argument("--kernel_size", type=int, default=3,
                        help='Kernel size for dilation')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help='Number of sampled rays per gradient step')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--sampling_strategy", type=str, default='random',
                        help='options: random / interest_point / interest_region')
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=0.0,
                        help='translation of camera (negative = zoom in)')
    # apply noise to observed image
    parser.add_argument("--noise", type=str, default='None',
                        help='options: gauss / salt / pepper / sp / poisson')
    parser.add_argument("--sigma", type=float, default=0.01,
                        help='var = sigma^2 of applied noise (variance = std)')
    parser.add_argument("--amount", type=float, default=0.05,
                        help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
    parser.add_argument("--delta_brightness", type=float, default=0.0,
                        help='reduce/increase brightness of the observed image, value is in [-1...1]')
    
    parser.add_argument("--class_name", type=str, default='01Gorilla',
                        help='LEGO-3D anomaly class')
    
    

    return parser


def rot_psi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]])


def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])


def rot_phi(psi): return np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])


def trans_t(t): return np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])


def load_blender(data_dir, model_name, obs_img_num, half_res, white_bkgd, *kwargs):

    with open(os.path.join(data_dir + str(model_name) + "/obs_imgs/", 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    img_path = os.path.join(data_dir + str(model_name) +
                            "/obs_imgs/", frames[obs_img_num]['file_path'] + '.png')
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    H, W = img_rgba.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if white_bkgd:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    imageio.imwrite("horse.png",img_rgb)
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    obs_img_pose = np.array(
        frames[obs_img_num]['transform_matrix']).astype(np.float32)
    phi, theta, psi, t = kwargs
    start_pose = trans_t(t) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta /
                                                                  180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    # image of type uint8
    return img_rgb, [H, W, focal], start_pose, obs_img_pose


def load_blender_AD(data_dir, model_name, obs_img_num, half_res, white_bkgd, method,**kwargs):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms_train.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])+'.png'
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)

            imgs.append(img)
            poses.append(pose)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)

    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    img_path = os.path.join(data_dir, str(model_name),'anomaly',str(obs_img_num)+".png")
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    if white_bkgd and img_rgba.shape[-1]==4:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    index_best = 0
    score_best = 0.5
    initial_pose = np.zeros([4, 4])
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res
    
    # use lpips
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],img_rgb,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],img_rgb,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    # image of type uint8
    return img_rgb, [H, W, focal],initial_pose,score_best

def load_blender_ad(data_dir, model_name,  half_res, white_bkgd):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd and img.shape[-1]==4:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)
            imgs.append(img)
            poses.append(pose)
            
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res.astype(np.uint8)
    return imgs,[H, W, focal],poses

def find_nearest(imgs,obs_img,poses,method):
    # use lpips
    score_best=0.5
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],obs_img,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],obs_img,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    return initial_pose
        
def calculate_lpips(img1,img2,net='vgg',use_gpu=True):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net)
    img1 = lpips.im2tensor(img1)  # RGB image from [-1,1]
    img2 = lpips.im2tensor(img2)

    if use_gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()
    score = loss_fn.forward(img1, img2)
    return score

    
def resmaps_ssim(img_input,img_pred):
    score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
    return score,resmap

def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def resmaps_l1(imgs_input, imgs_pred):
    resmaps = np.abs(imgs_input - imgs_pred)
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def calculate_resmaps(img_input, img_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if img_input.shape[-1] == 3:
        img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_pred_gray = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    else:
        img_input_gray = img_input
        img_pred_gray = img_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(img_input_gray, img_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(img_input_gray, img_pred_gray)
    # if dtype == "uint8":
        # resmaps = img_as_ubyte(resmaps)
    return scores, resmaps
def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr
def bgr2rgb(img_bgr):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imwrite(title, img_bgr)
    # cv2.imshow(title, img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_POI(img_rgb, DEBUG=False):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected_points.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates


# Misc
def img2mse(x, y): return torch.mean((x - y) ** 2)
def MAPE(x,y):return torch.mean(torch.abs((x-y)/y))
def Relative_L2(x,y):return torch.mean(torch.abs((x-y)**2/y**2))
def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

# Load llff data

# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i,
                                [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]

    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, bds


def load_llff_data(data_dir, model_name, obs_img_num, *kwargs, factor=8, recenter=True, bd_factor=.75, spherify=False):
    # factor=8 downsamples original imgs by 8x
    poses, bds, imgs = _load_data(
        data_dir + str(model_name) + "/", factor=factor)
    print('Loaded', data_dir + str(model_name) + "/", bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, bds = spherify_poses(poses, bds)

    #images = images.astype(np.float32)
    images = np.asarray(images * 255, dtype=np.uint8)
    poses = poses.astype(np.float32)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    obs_img = images[obs_img_num]
    obs_img_pose = np.concatenate(
        (poses[obs_img_num], np.array([[0, 0, 0, 1.]])), axis=0)
    phi, theta, psi, t = kwargs
    start_pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180. *
                                                     np.pi) @ rot_psi(psi/180.*np.pi) @ trans_t(t) @ obs_img_pose
    return obs_img, hwf, start_pose, obs_img_pose, bds


def pose_retrieval(imgs,obs_img,poses):
    # Prepare model.
    model = load_model(pretrained_model='./retrieval/model/net_best.pth', use_gpu=True)

    # Extract database features.
    gallery_feature = extract_feature(model=model, imgs=imgs)

    # Query.
    query_image = transform_query_image(obs_img)

    # Extract query features.
    query_feature = extract_feature_query(model=model, img=query_image)

    # Sort.
    similarity, index = sort_img(query_feature, gallery_feature)

    return poses[index[0]]


def pose_retrieval_efficient(imgs,obs_img,poses):
    # Prepare model.
    model = load_model_efficient()

    # Extract database features.
    gallery_feature = extract_feature_efficient(model=model, imgs=imgs)

    # Extract query features.
    query_feature = extract_feature_query_efficient(model=model, img=obs_img)

    # Sort.
    similarity, index = sort_img_efficient(query_feature, gallery_feature)

    return poses[index]

def pose_retrieval_loftr(imgs,obs_img,poses):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("retrieval/model/indoor_ds_new.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        match_num=len(mconf)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]
        