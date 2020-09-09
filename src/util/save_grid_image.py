import numpy as np
from PIL import Image

def save_grid_image(grid, path, scale=4):
	n_rows = len(grid)
	n_cols = len(grid[0])

	im_shape = grid[0][0].shape
	im_h = im_shape[0]
	im_w = im_shape[1]
	im_c = im_shape[2]

	total_h = n_rows * im_h
	total_w = n_cols * im_w
	total_c = im_c
	im_out = np.zeros(shape=[total_h,total_w,total_c],dtype=np.uint8)

	for row,row_ims in enumerate(grid):
		for col,im in enumerate(row_ims):
			assert im.shape[0] == im_h
			assert im.shape[1] == im_w
			assert im.shape[2] == im_c
			r = row * im_h
			c = col * im_w
			im_out[r:r+im_h,c:c+im_w,:] = im

	print('saved to: {}'.format(path))
	im_out = Image.fromarray(im_out)
	im_out = im_out.resize((total_w*scale,total_h*scale,),resample=Image.NEAREST)
	im_out.save(path)
