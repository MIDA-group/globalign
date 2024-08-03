import numpy as np
import nibabel as nib

import globalign
import nd2cat
import time

import json
import re
import os


# "Default" values are rather arbitrarily picked from 'example.py'
def register(ref_image, flo_image, k=8, grid_angles=100, refinement_angles=32):
    print(f'Reference image shape: {ref_image.shape}, Floating image shape:{flo_image.shape}')
    
    # quantize both images into k levels
    Q_A = k
    Q_B = k
    print(f'Using k={Q_A} and k={Q_B} quantization levels for ref. and flo. images, respectively')

    quantized_ref_image = nd2cat.image2cat_kmeans(ref_image, Q_A)
    quantized_flo_image = nd2cat.image2cat_kmeans(flo_image, Q_B)
    print(f'Quantized ref. image shape: {ref_image.shape}, Quantized flo. image shape: {flo_image.shape}', flush=True)

    # use the whole images / no masking
    M_ref = np.ones(quantized_ref_image.shape, dtype='bool')
    M_flo = np.ones(quantized_flo_image.shape, dtype='bool')

    overlap = 0.5 # at least 50% overlap
    refinement_param = {'n': refinement_angles, 'max_angle': 360.0 / grid_angles}

    # enable GPU processing (requires CUDA)
    on_gpu = True

    # run globalign
    t1 = time.time()
    param = globalign.align_rigid_and_refine(quantized_ref_image, quantized_flo_image, M_ref, M_flo, Q_A, Q_B, grid_angles, 180.0, refinement_param=refinement_param, overlap=overlap, enable_partial_overlap=True, normalize_mi=False, on_gpu=on_gpu, save_maps=False, rng = np.random.default_rng(12345))
    t2 = time.time()

    print(f'Time elapsed:       {t2-t1:.4}s')
    print(f'Mutual information: {param[0][0]:.4f}')
    print(f'Rotation angle:     {param[0][1]:.4f} degrees')
    print(f'Translation:        {param[0][2:4]}')
    print(f'Center of rotation: {param[0][4:6]}', flush=True)

    return param


def align(ref,flo):
    print(f'\nAligning {ref} (reference),\n     and {flo} (floating).')

    #Find ID of ref and flo
    refID = re.search(r"_(\d+)_", ref).group(1)
    floID = re.search(r"_(\d+)_", flo).group(1)

    #Load images
    ref_image=nib.load(ref).get_fdata()
    flo_image=nib.load(flo).get_fdata().squeeze()


    #Run CMIF based image registration
    param = register(ref_image,flo_image)
    
    #More fine grained search; requires a bit more compute
    # param = register(ref_image,flo_image,k=16,grid_angles=128,refinement_angles=64)


    #Compute displacement field according to Learn2Reg instructions
    ny, nx = flo_image.shape[:2]
    grid=np.mgrid[0:ny,0:nx].astype(np.float64)
    grid=np.flip(grid, 0) #Y,X -> X,Y
    grid=np.moveaxis(grid, 0, -1) #[[x,y],X,Y] -> [X,Y,[x,y]]

    gridpoints=grid.reshape((-1,2)) #List of [x,y] points
    warpedpoints=globalign.warp_points_rigid(gridpoints, param[0]) #Warp the points (ref->flo)
    displace=warpedpoints.reshape(grid.shape)-grid #Subtract the identity grid to make displacement only
    displace=np.swapaxes(displace,0,1) #Empirically found to be needed

    displace=np.expand_dims(displace, axis=2) #Reshape to [h,w,1,2], i.e., [X,Y,1,[x,y]]
    nib.save(nib.Nifti1Image(displace,None),f'output/disp_{refID}_{floID}.nii.gz')


if __name__ == '__main__':
    BASE_PATH='COMULISSHGBF'
    os.mkdir('output') #Intentionally fail if the directory exists, to not risk overwriting

    #Parse filenames from COMULISSHGBF_dataset.json
    with open(os.path.join(BASE_PATH,'COMULISSHGBF_dataset.json')) as f:
        d = json.load(f)

        #Run registration of validation data
        for x in d['registration_val']:
            align(os.path.join(BASE_PATH,x['fixed']), os.path.join(BASE_PATH,x['moving']))
