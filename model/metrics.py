import numpy as np

def tensor2im(image_tensor, imtype=np.float32, min_max=(-1, 1)):
    image_numpy = image_tensor.squeeze().cpu().float().numpy()
    image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = len(image_numpy.shape)
    if n_dim == 4:
        nc, nd, nh, nw = image_numpy.shape
        image_numpy = np.transpose(image_numpy[:, int(nd / 2)], (1, 2, 0))
        image_numpy -= np.amin(image_numpy)
        image_numpy /= np.amax(image_numpy)
    elif n_dim == 3:
        nc, nh, nw = image_numpy.shape
        tmp = np.zeros((nh, nw, 3))
        tmp[:, :, :2] = image_numpy.transpose(1, 2, 0)
        image_numpy = tmp
        image_numpy -= np.amin(image_numpy)
        image_numpy = (image_numpy / np.amax(image_numpy))
    elif n_dim == 2:
        nh, nw = image_numpy.shape
        image_numpy = image_numpy.reshape(nh, nw, 1)
        image_numpy = np.tile(image_numpy, (1, 1, 3))

    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)

