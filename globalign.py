
###
### Multimodal registration with exhaustive search mutual information
### Author: Johan \"{O}fverstedt
###

import torch
import numpy as np
import torch.nn.functional as F
import torch.fft
import torchvision.transforms.functional as TF
import transformations

VALUE_TYPE = torch.float32

# Creates a list of random angles
def grid_angles(center, radius, n = 32):
    angles = []

    n_denom = n
    if radius < 180:
        n_denom -= 1

    for i in range(n):
        i_frac = i/n_denom
        ang = center + (2.0 * i_frac - 1.0) * radius
        angles.append(ang)

    return angles

# Supply a Random number generator (e.g. 'rng = np.random.default_rng(12345)') for reproducible results
# Default: 'rng=None' -> np.random.default_rng()
def random_angles(centers, center_prob, radius, n = 32, rng = None):
    if rng is None: 
        rng = np.random.default_rng()

    angles = []

    if not isinstance(centers, list):
        centers = [centers]
    if center_prob is not None:
        mass = np.sum(center_prob)
        p = center_prob / mass
    else:
        p = None

    for i in range(n):
        c = rng.choice(centers, p=p, replace=True)
        frac = rng.random()
        ang = c + (2.0 * frac - 1.0) * radius

        angles.append(ang)

    return angles

### Helper functions

def compute_entropy(C, N, eps=1e-7):
    p = C/N
    return p*torch.log2(torch.clamp(p, min=eps, max=None))

def float_compare(A, c):
    return torch.clamp(1-torch.abs(A-c), 0.0)

def fft_of_levelsets(A, Q, packing, setup_fn):
    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = np.minimum(a_start + packing, Q)
        levelsets = []
        for a in range(a_start, a_end):
            levelsets.append(float_compare(A, a))
        A_cat = torch.cat(levelsets, 0)
        del levelsets
        ffts = setup_fn(A_cat)
        del A_cat
        fft_list.append((ffts, a_start, a_end))
    return fft_list

def fft(A):
    spectrum = torch.fft.rfft2(A)
    return spectrum

def ifft(Afft):
    res = torch.fft.irfft2(Afft)
    return res

def fftconv(A, B):
    C = A * B
    return C

def corr_target_setup(A):
    B = fft(A)
    return B

def corr_template_setup(B):
    B_FFT = torch.conj(fft(B))

    return B_FFT

def corr_apply(A, B, sz, do_rounding = True):
    C = fftconv(A, B)

    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3]]

    if do_rounding:
        C = torch.round(C)

    return C

def tf_rotate(I, angle, fill_value, center=None):
    if center is not None:
        center = [x+0.5 for x in center] # Half a pixel offset, since TF.rotate origin is in upper left corner
    return TF.rotate(I, -angle, center=center, fill=[fill_value, ])

def create_float_tensor(shape, on_gpu, fill_value=None):
    if on_gpu:
        res = torch.empty((shape[0], shape[1], shape[2], shape[3]), device='cuda', dtype=torch.float32)
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full((shape[0], shape[1], shape[2], shape[3]), fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

def to_tensor(A, on_gpu=True):
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)


### End helper functions

###
### align_rigid
###
### Performs rigid alignment of multimodal images using exhaustive search mutual information (MI),
### locating the global maximum of the MI measure w.r.t. all possible whole-pixel translations as well
### as a set of enumerated rotations. Runs on the GPU, using PyTorch.
###
### Parameters:
### A: (reference 2d image).
### B: (floating 2d image).
### M_A: (reference 2d mask image).
### M_B: (floating 2d mask image).
### Q_A: (number of quantization levels in image A).
### Q_B: (number of quantization levels in image B).
### angles: List of angles for the rigid alignment.
### overlap: The required overlap fraction (of the maximum overlap possible, given masks).
### enable_partial_overlap: If False then no padding will be done, and only fully overlapping
###    configurations will be evaluated. If True, then padding will be done to include
###    configurations where only part of image B is overlapping image A.
### normalize_mi: Flag to choose between normalized mutual information (NMI) or
###    standard unnormalized mutual information.
### on_gpu: Flag controlling if the alignment is done on the GPU.
### save_maps: Flag for exporting the stack of CMIF maps over the angles, e.g. for debugging or visualization.
### Returns: np.array with 6 values (mutual_information, angle, y, x, y of center of rotation (origin at center of top left pixel), x of center of rotation), maps/None.
###
###  Note: Prior to v1.0.2, the returned center of rotation used Torchvisions convention 'Origin is the upper left corner' which
###   is incompatible with the followed use of 'scipy.ndimage.interpolation.map_coordinates' that assumes integer coordinate-centered pixels.
###
def align_rigid(A, B, M_A, M_B, Q_A, Q_B, angles, overlap=0.5, enable_partial_overlap=True, normalize_mi=False, on_gpu=True, save_maps=False):
    eps=1e-7

    results = []
    maps = []

    A_tensor = to_tensor(A, on_gpu=on_gpu)
    B_tensor = to_tensor(B, on_gpu=on_gpu)

    if A_tensor.shape[-1] < 1024:
        packing = np.minimum(Q_B, 64)
    elif A_tensor.shape[-1] <= 2048:
        packing = np.minimum(Q_B, 8)
    elif A_tensor.shape[-1] <= 4096:
        packing = np.minimum(Q_B, 4)
    else:
        packing = np.minimum(Q_B, 1)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = to_tensor(M_A, on_gpu)
        A_tensor = torch.round(M_A * A_tensor + (1-M_A) * (Q_A+1))
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = to_tensor(M_B, on_gpu)
        
    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_sz = (round(B.shape[-1]*(1.0-overlap)), round(B.shape[-2]*(1.0-overlap)))
        A_tensor = F.pad(A_tensor, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=Q_A+1)
        M_A = F.pad(M_A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0], partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=0)
    else:
        partial_overlap_pad_sz = (0, 0)

    ext_ashape = A_tensor.shape
    ashape = ext_ashape[2:]
    ext_bshape = B_tensor.shape
    bshape = ext_bshape[2:]
    b_pad_shape = torch.tensor(A_tensor.shape, dtype=torch.long)-torch.tensor(B_tensor.shape, dtype=torch.long)
    ext_valid_shape = b_pad_shape + 1
    batched_valid_shape = ext_valid_shape + torch.tensor([packing-1, 0, 0, 0])
    valid_shape = ext_valid_shape[2:]

    # use default center of rotation (which is the center point)
    center_of_rotation = transformations.image_center_point(B)

    M_A_FFT = corr_target_setup(M_A)

    A_ffts = []
    for a in range(Q_A):
        A_ffts.append(corr_target_setup(float_compare(A_tensor, a)))

    del A_tensor
    del M_A

    if normalize_mi:
        H_MARG = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = create_float_tensor(ext_valid_shape, on_gpu, 0.0)

    for ang in angles:

        # preprocess B for angle
        B_tensor_rotated = tf_rotate(B_tensor, ang, Q_B, center=center_of_rotation)

        M_B_rotated = tf_rotate(M_B, ang, 0, center=center_of_rotation)
        B_tensor_rotated = torch.round(M_B_rotated * B_tensor_rotated + (1-M_B_rotated) * (Q_B+1))
        B_tensor_rotated = F.pad(B_tensor_rotated, (0, ext_ashape[-1]-ext_bshape[-1], 0, ext_ashape[-2]-ext_bshape[-2], 0, 0, 0, 0), mode='constant', value=Q_B+1)
        M_B_rotated = F.pad(M_B_rotated, (0, ext_ashape[-1]-ext_bshape[-1], 0, ext_ashape[-2]-ext_bshape[-2], 0, 0, 0, 0), mode='constant', value=0)

        M_B_FFT = corr_template_setup(M_B_rotated)
        del M_B_rotated

        N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape), min=eps, max=None)

        b_ffts = fft_of_levelsets(B_tensor_rotated, Q_B, packing, corr_template_setup)

        for bext in range(len(b_ffts)):
            b_fft = b_ffts[bext]
            E_M = torch.sum(compute_entropy(corr_apply(M_A_FFT, b_fft[0], batched_valid_shape), N, eps), dim=0)
            if normalize_mi:
                H_MARG = torch.sub(H_MARG, E_M)
            else:
                MI = torch.sub(MI, E_M)
            del E_M

            for a in range(Q_A):
                A_fft_cuda = A_ffts[a]
                    
                if bext == 0:
                    E_M = compute_entropy(corr_apply(A_fft_cuda, M_B_FFT, ext_valid_shape), N, eps)
                    if normalize_mi:
                        H_MARG = torch.sub(H_MARG, E_M)
                    else:
                        MI = torch.sub(MI, E_M)
                    del E_M
                E_J = torch.sum(compute_entropy(corr_apply(A_fft_cuda, b_fft[0], batched_valid_shape), N, eps), dim=0)
                if normalize_mi:
                    H_AB = torch.sub(H_AB, E_J)
                else:
                    MI = torch.add(MI, E_J)
                del E_J
                del A_fft_cuda
            del b_fft
            if bext == 0:
                del M_B_FFT

        del B_tensor_rotated

        if normalize_mi:
            MI = torch.clamp((H_MARG / (H_AB + eps) - 1), 0.0, 1.0)
            
        if save_maps:
            maps.append(MI.cpu().numpy())

        (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
        N_filt = torch.lt(N, overlap*max_n)
        MI[N_filt] = 0.0
        del N_filt, N

        MI_vec = torch.reshape(MI, (-1,))
        (val, ind) = torch.max(MI_vec, -1)

        results.append((ang, val, ind))

        if normalize_mi:
            H_MARG.fill_(0)
            H_AB.fill_(0)
        else:
            MI.fill_(0)


    print('------------------------------')
    print(' [MI]   [angle]  [dx] [dy] ')
    cpu_results = []
    for i in range(len(results)):
        ang = results[i][0]
        maxval = results[i][1].cpu().numpy()
        maxind = results[i][2].cpu().numpy()
        sz_x = int(ext_valid_shape[3].numpy())
        y = maxind // sz_x
        x = maxind % sz_x
        cpu_results.append((maxval, ang, -(y - partial_overlap_pad_sz[1]), -(x - partial_overlap_pad_sz[0]), center_of_rotation[1], center_of_rotation[0]))
    cpu_results = sorted(cpu_results, key=(lambda tup: tup[0]), reverse=True)
    for i in range(len(cpu_results)):
        res = cpu_results[i]
        print('%.4f %8.3f %4d %4d' %(float(res[0]), res[1], res[2], res[3]))
    print('------------------------------')
    # Return the maximum found
    if save_maps:
        return cpu_results, maps
    else:
        return cpu_results, None

###
### align_rigid_and_refine
###
### Performs rigid alignment of multimodal images using exhaustive search mutual information (MI),
### locating the global maximum of the MI measure w.r.t. all possible whole-pixel translations as well
### as a set of enumerated rotations. Runs on the GPU, using PyTorch.
###
### Parameters:
### A: (reference 2d image).
### B: (floating 2d image).
### M_A: (reference 2d mask image).
### M_B: (floating 2d mask image).
### Q_A: (number of quantization levels in image A).
### Q_B: (number of quantization levels in image B).
### angles_n: Number of angles to consider in the grid search.
### max_angle: The largest angle to include in the grid search. (180 => global search)
### refinement_param: dictionary with settings for the refinement steps e.g. {'n': 32, 'max_angle': 3.0}
### overlap: The required overlap fraction (of the maximum overlap possible, given masks).
### enable_partial_overlap: If False then no padding will be done, and only fully overlapping
###    configurations will be evaluated. If True, then padding will be done to include
###    configurations where only part of image B is overlapping image A.
### normalize_mi: Flag to choose between normalized mutual information (NMI) or
###    standard unnormalized mutual information.
### on_gpu: Flag controlling if the alignment is done on the GPU.
### save_maps: Flag for exporting the stack of CMIF maps over the angles, e.g. for debugging or visualization.
### rng: Optional random number generator (e.g. 'rng = np.random.default_rng(12345)') for reproducible results; default: None -> np.random.default_rng()
### Returns: np.array with 6 values (mutual_information, angle, y, x, y of center of rotation, x of center of rotation), maps/None.
###
def align_rigid_and_refine(A, B, M_A, M_B, Q_A, Q_B, angles_n, max_angle, refinement_param={'n': 32}, overlap=0.5, enable_partial_overlap=True, normalize_mi=False, on_gpu=True, save_maps=False, rng=None):
    angles1 = grid_angles(0, max_angle, n=angles_n)
    param, maps1 = align_rigid(A, B, M_A, M_B, Q_A, Q_B, angles1, overlap, enable_partial_overlap, normalize_mi, on_gpu, save_maps=save_maps)
    # extract rotations and probabilities for refinement
    centers = []
    center_probs = []
    for i in range(np.minimum(1, len(param))):
        par = param[i]
        centers.append(par[1])
        center_probs.append(par[0])
    # param[0] now the current optimimum
    rot = param[1]
    if refinement_param.get('n', 0) > 0:
        angles2 = random_angles(centers, center_probs, refinement_param.get('max_angle', 3.0), n = refinement_param.get('n'), rng = rng)
        param2, maps2 = align_rigid(A, B, M_A, M_B, Q_A, Q_B, angles2, overlap, enable_partial_overlap, normalize_mi, on_gpu, save_maps=save_maps)
        param = param + param2
        param = sorted(param, key=(lambda tup: tup[0]), reverse=True)
        return np.array(param[0]), (maps1, maps2)
    else:
        return np.array(param[0]), (maps1,)

###
### warp_image_rigid
###
### Applies the transformation obtained by the functions align_rigid/align_rigid_and_refine
### to warp a floating image into the space of the ref_image (using backward mapping).
### param: The parameters (first value of the returned tuple from align_rigid/align_rigid_and_refine)
### mode (interpolation): nearest/linear/spline
### bg_value: The value to insert where there is no information in the flo_image
### inv: Invert the transformation, used e.g. when warping the original reference image into
###      the space of the original floating image.
def warp_image_rigid(ref_image, flo_image, param, mode='nearest', bg_value=0.0, inv=False):
    r = transformations.Rotate2DTransform()
    r.set_param(0, np.pi*param[1]/180.0)
    translation = transformations.TranslationTransform(2)
    translation.set_param(0, param[2])
    translation.set_param(1, param[3])
    t = transformations.CompositeTransform(2, [translation, r])

    t = transformations.make_centered_transform(t, np.array(param[4:]), np.array(param[4:]))
    if inv:
        t = t.invert()

    out_shape = ref_image.shape[:2] + flo_image.shape[2:]
    flo_image_out = np.zeros(out_shape, dtype=flo_image.dtype)
    if flo_image.ndim == 3:
        for i in range(flo_image.shape[2]):
            bg_val_i = np.array(bg_value)
            if bg_val_i.shape[0] == flo_image.shape[2]:
                bg_val_i = bg_val_i[i]

            t.warp(flo_image[:, :, i], flo_image_out[:, :, i], in_spacing=np.ones(2,), out_spacing=np.ones(2,), mode=mode, bg_value=bg_val_i)
    else:
        t.warp(flo_image, flo_image_out, in_spacing=np.ones(2,), out_spacing=np.ones(2,), mode=mode, bg_value=bg_value)

    return flo_image_out

###
### warp_points_rigid
###
### Applies the transformation obtained by the functions align_rigid/align_rigid_and_refine
### to transform a set of points in the reference image space into the floating image space.
### param: The parameters (first value of the returned tuple from align_rigid/align_rigid_and_refine)
### inv: Invert the transformation, used e.g. when transforming points from the original floating image
###      space into the original reference image space.
def warp_points_rigid(points, param, inv=False):
    r = transformations.Rotate2DTransform()
    r.set_param(0, np.pi*param[1]/180.0)
    translation = transformations.TranslationTransform(2)
    translation.set_param(0, param[2])
    translation.set_param(1, param[3])
    t = transformations.CompositeTransform(2, [translation, r])

    t = transformations.make_centered_transform(t, np.array(param[4:]), np.array(param[4:]))
    if inv:
        t = t.invert()
    
    return t.transform(points)
