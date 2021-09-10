
import numpy as np

import skimage
from skimage.util import img_as_ubyte
import skimage.io as io

import globalign
import nd2cat

import time

def draw_matching_squares(im1, im2, pos, sz, color_table, c1):
    for c in range(color_table[0].shape[0]):
        im1[pos[0]:pos[0]+sz[0], pos[1]:pos[1]+sz[1], c] = color_table[c1][c]
        im2[pos[0]:pos[0]+sz[0], pos[1]:pos[1]+sz[1], c] = color_table[len(color_table)-1-c1][c]

def example():
    sz = 512
    sq_sz = 32

    ref_image = np.zeros((sz, sz, 3))
    flo_image = np.ones((sz, sz, 3))

    # color count: 6
    # make color table
    colors = []
    for i in range(3):
        c = np.array([0.0, 0.0, 0.0])
        c[i] += 1.0
        colors.append(c)

    for i in range(2):
        for j in range(i+1, 3):
            c = np.array([0.0, 0.0, 0.0])
            c[i] += 1.0
            c[j] += 1.0
            colors.append(c)

    draw_matching_squares(ref_image, flo_image, [20, 20], [30, 30], colors, 0)
    draw_matching_squares(ref_image, flo_image, [60, 20], [30, 30], colors, 1)
    draw_matching_squares(ref_image, flo_image, [30, 100], [20, 20], colors, 2)
    draw_matching_squares(ref_image, flo_image, [200, 250], [40, 40], colors, 3)
    draw_matching_squares(ref_image, flo_image, [250, 200], [40, 40], colors, 4)
    draw_matching_squares(ref_image, flo_image, [400, 150], [70, 40], colors, 5)

    draw_matching_squares(ref_image, flo_image, [400, 20], [30, 30], colors, 0)
    draw_matching_squares(ref_image, flo_image, [20, 400], [30, 30], colors, 1)
    draw_matching_squares(ref_image, flo_image, [400, 400], [20, 20], colors, 2)
    draw_matching_squares(ref_image, flo_image, [100, 350], [40, 30], colors, 3)
    draw_matching_squares(ref_image, flo_image, [220, 280], [50, 40], colors, 4)
    draw_matching_squares(ref_image, flo_image, [480, 277], [20, 40], colors, 5)

    flo_image_rot = np.rot90(flo_image, 1, (0, 1))
    flo_image_rot = np.roll(flo_image_rot, shift=(17, -12), axis=(0, 1))

    io.imsave('example_ref_image.png', img_as_ubyte(ref_image))
    io.imsave('example_flo_image.png', img_as_ubyte(flo_image))
    io.imsave('example_flo_image_rot.png', img_as_ubyte(flo_image_rot))

    # quantize (k=8)

    Q_A = 8
    Q_B = 8
    
    quantized_ref_image = nd2cat.image2cat_kmeans(ref_image, Q_A)
    quantized_flo_image_rot = nd2cat.image2cat_kmeans(flo_image_rot, Q_B)
    M_ref = np.ones((sz, sz), dtype='bool')
    M_flo = np.ones((sz, sz), dtype='bool')
    overlap = 0.5
    grid_angles = 100
    refinement_param = {'n': 32, 'max_angle': 360.0 / grid_angles}
    
    # align

    # enable GPU processing (requires CUDA)
    on_gpu = True

    t1 = time.time()

    param = globalign.align_rigid_and_refine(quantized_ref_image, quantized_flo_image_rot, M_ref, M_flo, Q_A, Q_B, grid_angles, 180.0, refinement_param=refinement_param, overlap=overlap, enable_partial_overlap=True, normalize_mi=False, on_gpu=on_gpu, save_maps=False)

    t2 = time.time()
    print('Time elapsed: ', t2-t1)
    print('Mutual information: ', param[0][0])
    print('Rotation angle: ', param[0][1])
    print('Translation: ', param[0][2:4])
    print('Center of rotation: ', param[0][4:6])

    # apply parameters to the floating image

    flo_image_recovered = globalign.warp_image_rigid(ref_image, flo_image_rot, param[0], mode='nearest', bg_value=[1.0, 1.0, 1.0])

    io.imsave('example_flo_image_recovered.png', img_as_ubyte(flo_image_recovered))

if __name__ == '__main__':
    example()