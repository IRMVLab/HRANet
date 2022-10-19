from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import *
from torch.autograd import Variable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)

def euler2quat(rot):
    '''
    :param rot: [B,3]
    '''
    cy = torch.cos(rot[:, 0] * 0.5)
    sy = torch.sin(rot[:, 0] * 0.5)
    cp = torch.cos(rot[:, 1] * 0.5)
    sp = torch.sin(rot[:, 1] * 0.5)
    cr = torch.cos(rot[:, 2] * 0.5)
    sr = torch.sin(rot[:, 2] * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    quat = torch.stack((w, x, y, z), dim=1)
    return quat


def augmentation_loss(args, aug_pose, poses, w_t, w_r):
    '''
    input:
    b: batch_size
    aug_pose:  random pose
    poses:  computed pose from the network
    '''
    aug_loss = 0.0
    for pose in poses:
        for i in range(len(pose)): # pose hierarchy
            diff_p = pose[i] - aug_pose
            loss_t = torch.mean(torch.sqrt(diff_p[:,0:3] * diff_p[:,0:3] + 1e-10))
            if args.augloss_mode =='euler':
                loss_r = torch.mean(torch.sqrt(torch.sum(diff_p[:,3:]*diff_p[:,3:], axis=-1, keepdim=True) + 1e-10))
            elif args.augloss_mode =='quat':
                quat1 = euler2quat(pose[i][:, 3:])
                quat2 = euler2quat(aug_pose[:, 3:])
                loss_r = torch.mean(torch.sqrt(torch.sum((quat1 - quat2) * (quat1 - quat2), axis=-1, keepdim=True) + 1e-10))
            aug_loss += loss_t * torch.exp(-w_t) + w_t + loss_r * torch.exp(-w_r) + w_r

    return aug_loss

def compute_aug_loss_inv(args,poses,poses_inv,T_aug, w_t, w_r):
    loss1 = augmentation_loss(args, T_aug , poses, w_t, w_r)
    loss2 = augmentation_loss(args, -T_aug, poses_inv, w_t,w_r)
    return loss1+loss2

# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode):

    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth,  pose_list, pose_inv_list in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):

            # upsample depth
            for i in range(len(pose_list)):

                pose = pose_list[i]
                pose_inv = pose_inv_list[i]
                b, _, h, w = tgt_img.size()
                tgt_img_scaled = tgt_img
                ref_img_scaled = ref_img
                intrinsic_scaled = intrinsics
                if s == 0:
                    tgt_depth_scaled = tgt_depth[s]
                    ref_depth_scaled = ref_depth[s]
                else:
                    tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='nearest')
                    ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='nearest')
    
                if i == 0: ## pose2
                    photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                        intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
                    photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                        intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
                else:  ## pose1
                    photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled.detach(), ref_depth_scaled.detach(), pose,
                                                                        intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
                    photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled.detach(), tgt_depth_scaled.detach(), pose_inv,
                                                                        intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
    
                photo_loss += ((photo_loss1 + photo_loss2))
                geometry_loss += ((geometry_loss1 + geometry_loss2))

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose,  intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode):

    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        # mean_disp = disp.mean(2, True).mean(3, True)
        # norm_disp = disp / (mean_disp + 1e-7)
        max_disp = disp.max(3)[0].unsqueeze(3).max(2)[0].unsqueeze(2)
        norm_disp = disp / (max_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

@torch.no_grad()
def test_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

        rmse = (current_gt - current_pred) ** 2
        rmse = torch.sqrt(rmse.mean())
        rmse_log = (torch.log(current_gt) - torch.log(current_pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.mean())
    
    return [metric.item() / batch_size for metric in [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]
