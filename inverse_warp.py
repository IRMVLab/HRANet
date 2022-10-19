from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage.filters import gaussian, sobel
from scipy.interpolate import griddata
from ctypes import *
import ctypes
import cv2
            

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combination of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2


    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode)

    return projected_img, valid_mask, projected_depth, computed_depth


def inverse_warp3(img, depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane in the model
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')


    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]


    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, _ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    return projected_img

def Merge(pose_c, pose_d):
    mat_c = euler2mat(pose_c[:, 3:])

    mat_new = torch.matmul(mat_c, pose_d[:, :3].reshape(-1, 3, 1)).squeeze(-1)

    eu_angle = pose_c[:, 3:] + pose_d[:, 3:]

    t_mat = mat_new + pose_c[:, :3]

    return torch.cat((t_mat, eu_angle), 1)

def inverse_warp4(img, depth, pose, T_aug, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane in the model
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose2= Merge(pose,T_aug)
    pose_mat = pose_vec2mat(pose2)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, _ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    return projected_img



def sharp(depth, use_filter=False):
    disp = 1 / depth

    b, _, h, w = disp.size()
    sdisp = torch.zeros_like(depth)
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(disp).cpu().detach().numpy()  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(disp).cpu().detach().numpy()  # [1, H, W]

    for i in range(b):
        adisp = disp[i]
        dispr = adisp / adisp.max()
        dispr *= 50
        dispr = dispr.cpu().detach().numpy()[0]
        edges = sobel(dispr) > 3
        dispr[edges] = 0
        mask = dispr > 0
        mask = np.expand_dims(mask, axis=0)

        adisp = griddata(np.stack([i_range[mask].ravel(), j_range[mask].ravel()], 1),
                         adisp.cpu().detach().numpy()[mask].ravel(), np.stack([i_range.ravel(),
                                                                               j_range.ravel()], 1),
                         method='nearest').reshape(1, h, w)
        
        if use_filter:
            adisp=cv2.medianBlur(adisp[0],5)

        sdisp[i] = torch.from_numpy(np.expand_dims(adisp,0)).type_as(depth)

    return 1 / sdisp

def img_aug2(img, depth, intrinsics, args):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """

    batch_size, _, img_height, img_width = img.size()

    depth = sharp(depth)

    aug_imgs = {}

    t = args.aug_rot
    anglex = np.clip(t * 0.005 * np.random.randn(batch_size, 1), -0.01 * t, t * 0.01).astype(
        np.float32) * np.pi / 4.0  # pitch
    angley = np.clip(t * 0.025 * np.random.randn(batch_size, 1), -0.05 * t, t * 0.05).astype(
        np.float32) * np.pi / 4.0  # yaw
    anglez = np.clip(t * 0.005 * np.random.randn(batch_size, 1), -0.01 * t, t * 0.01).astype(
        np.float32) * np.pi / 4.0  # roll

    t2 = args.aug_trans      
    tx = np.clip(t2 * 0.05 * np.random.randn(batch_size, 1), -0.1 * t2, t2 * 0.1).astype(np.float32)  # left_right
    ty = np.clip(t2 * 0.025 * np.random.randn(batch_size, 1), -0.075 * t2, t2 * 0.075).astype(np.float32)  # up_down
    tz = np.clip(t2 * 0.25 * np.random.randn(batch_size, 1), -0.5 * t2, t2 * 0.5).astype(np.float32)  # forward_back

    pose = torch.from_numpy(np.concatenate([tx, ty, tz, anglex, angley, anglez], axis=1)).to(depth.get_device())

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode='None')  # [B,H,W,2]

    lib = cdll.LoadLibrary("external/forward_warping/libwarping.so")
    warp = lib.forward_warping

    aug_img = torch.zeros_like(img).to(img.device)

    p1 = src_pixel_coords.cpu().detach()
    p1 = (p1 + 1) / 2 ## zeros: 1.5
    p1[:, :, :, 0] *= img_width - 1
    p1[:, :, :, 1] *= img_height - 1

    safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), img_height - 1), 0)
    safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), img_width - 1), 0)

    for i in range(batch_size):
        subangle = angley[i]
        subtrans = tx[i]
        oof_left = subangle > 0 or subtrans > 0

        warped_arr = np.zeros(img_height * img_width * 5).astype(np.uint8)
        warped_arr_depth = np.zeros(img_height * img_width * 5).astype(np.uint8)
        z1 = computed_depth[i].cpu().detach().numpy() 
        z1 = z1 / z1.max()
        subimg = (img[i].cpu().detach().numpy().transpose(1, 2, 0) * 0.225 + 0.45) * 255  ## use raw rgb value
        
        subimg = subimg.astype(np.uint8)
        subimg = subimg.reshape(-1)

        warp(c_void_p(subimg.ctypes.data), c_void_p(safe_x[i].numpy().ctypes.data),
             c_void_p(safe_y[i].numpy().ctypes.data), c_void_p(z1.reshape(-1).ctypes.data),
             c_void_p(warped_arr.ctypes.data), c_int(img_height), c_int(img_width))
        warped_arr = warped_arr.reshape(1, img_height, img_width, 5).astype(np.uint8)
        im1_raw = warped_arr[0, :, :, 0:3]

        masks = {}
        masks["H"] = warped_arr[0, :, :, 3:4]

        # Collision mask M
        masks["M"] = warped_arr[0, :, :, 4:5]
        # Keep all pixels that are invalid (H) or collide (M)
        masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)

        # Dilated collision mask M'
        kernel = np.ones((3, 3), np.uint8)
        masks["M'"] = cv2.dilate(masks["M"], kernel, iterations=1)
        masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)

        # Final mask P
        masks["H'"] = masks["H"] * masks["P"]

        kernel2 = np.ones((4, 4), np.uint8)
        mask_dilated = cv2.dilate(masks["H'"], kernel2, 4)

        MASK = np.expand_dims(mask_dilated, 2) - masks["H'"]
        im1 = cv2.inpaint(im1_raw, MASK, 3, cv2.INPAINT_TELEA)
        mask_dilated=np.stack((mask_dilated,mask_dilated,mask_dilated), axis=2)
        if (args.aug_padding == 'gauss'):
            gauss = (np.random.randn(img_height, img_width, 3).clip(-1, 1) + 1) * 255 / 2
            im1[mask_dilated == 0] = gauss[mask_dilated == 0]
        else:
            im1[mask_dilated == 0] = 0

        ctypes._reset_cache()

        im1 = ((im1.astype(np.float32) / 255) - 0.45) / 0.225
        im1 = torch.from_numpy(im1.transpose(2, 0, 1)).float().to(img.device)

        aug_img[i] = im1

    aug_imgs["aug_img"] = aug_img

    return aug_imgs, pose

def mat2euler(M, cy_thresh=None, seq='zyx'):
    
    m = M.cpu().detach().numpy()
    if cy_thresh is None:
        cy_thresh = np.finfo(m.dtype).eps * 4

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.reshape(9,-1)
    
    cy = torch.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: 
            z = torch.atan2(-r12,  r11) 
            y = torch.atan2(r13,  cy) 
            x = torch.atan2(-r23, r33) 
        else: 
            z = torch.atan2(r21,  r22)
            y = torch.atan2(r13,  cy) 
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = torch.atan2(-r31, cy)
            x = torch.atan2(r32, r33)
            z = torch.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = torch.atan2(r12, r13)  
            else:
                y = -np.pi/2
                
    else:
        raise Exception('Sequence not recognized')
    return torch.cat((x, y, z),0)
    
def mat_euler(M):

    euler = torch.cat(([mat2euler(M[i]).unsqueeze(0) for i in range(M.size(0))]),0)
    
    return euler
    
def mat2pose(mat_new):

    rot_mat = mat_new[:,:3,:3]
    eu_angle = mat_euler(rot_mat).reshape(-1,3)
    t_mat = mat_new[:,:3,3:].reshape(-1,3)
    
    return torch.cat((eu_angle,t_mat),1)
    
def pose_vec2mat_revised(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    b, _, _ = transform_mat.size()
    filler = Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])).type_as(transform_mat).expand(b, 1, 4)

    transform_mat = torch.cat([transform_mat, filler], dim=1)# [B, 4, 4]

    return transform_mat
    