#!/usr/bin/env python3

"""Image Encoding with PCA and other dimension reduction methods

Ref.
M. A. Turk A. P. Pentland. Face Recognition Using Eigenfaces, 1991.

Requirements:

numpy
sklearn
pillow
(optional) PIL_ext (an extension of pillow, created by the author)

Example:
    ```
    def demo(folder=pathlib.Path.cwd(), *args, **kwargs):
        # demo for image processing by PCA/ICA
        # save images in the folder before demo!
        
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        # define a model, such as PCA, NMF
        model = PCA(*args, **kwargs)
        ip = ImageEncoder(model)
        # a user-friendly API calling `fit` method of the model
        ip.ezfit(folder=folder)
        # save the eigen images
        for k, im in enumerate(ip.eigen_images):
            save_folder = folder / 'eigen'
            save_folder.mkdir(exist_ok=True)
            im.save(save_folder / f'{k}.jpg')
        # generate artificial images
        for k, im in enumerate(ip.generate(10, toimage=True)):
            im.save(save_folder / f'artificial{k}.jpg')
   ```
"""

# from builtins import isinstance
import pathlib

import numpy as np
import numpy.linalg as LA

from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

import scipy.special as ss


# Following are helpers for transforming images to vecters
def image2vector(image, channel=None):
    # image -> vector
    data = np.asarray(image, dtype=np.float64)
    if channel:
        return data[:,:, channel].ravel()
    else:
        return data.ravel()


def images2matrix(images, as_row=True, size=None):

    # images -> matrix
    if size:
        matrix = [image2vector(image.resize(size)) for image in images]
    else:
        matrix = [image2vector(image) for image in images]
    if as_row:
        return np.row_stack(matrix)
    else:
        return np.column_stack(matrix)


def vector2image(vector, size, mode='RGB'):
    # vector -> image
    if mode == 'RGB':
        if len(size)==2:
            size = (*size, 3)
    elif mode == 'RGBA':
        if len(size)==2:
            size = (*size, 4)

    return Image.fromarray(vector.reshape(size).astype('uint8')).convert(mode)


def get_image_names(files=None, folder=None, exts=('.jpg','.jpeg','.png')):
    """get names of images from files or a folder
    
    Keyword Arguments:
        files {List[Path]} -- jpg or jpeg files (default: {None})
        folder {Path} -- the folder of images (default: {None})
        exts {tuple[str]} -- (default: {('.jpg','.jpeg','.png')})
    
    Returns:
        List[Image] -- list of images
    
    Raises:
        Exception -- Provide files or a folder
        LookupError -- A file name is invalid
    """
    images = []
    if files:
        if folder:
            files += [f for f in pathlib.Path(folder).iterdir()]
    elif folder:
        files = pathlib.Path(folder).iterdir()
    else:
        raise Exception('Must provide files or a folder')

    for f in files:
        if isinstance(f, str):
            f = pathlib.Path(f)
        if f.suffix == '':
            for ext in exts:
                f = pathlib.Path(f).with_suffix(ext)
                if f.exists():
                    images.append(str(f))
                    break
        elif f.exists() and f.suffix in exts:
            images.append(str(f))
        else:
            raise LookupError('Invalid file name %s' % f)

    return images

def get_images(*args, **kwargs):
    return map(Image.open, get_image_names(*args, **kwargs))

# def op_images(m):
#     def mm(obj, images, *args, **kwargs):
#         with Image.open(images[0]) as image:
#             size = image.size
#             mode = image.mode
#         X = images2matrix(images, size=size)
#         if obj.size is None:
#             obj.size = size
#         if obj.mode is None:
#             obj.mode = mode
#         return m(obj, X, *args, **kwargs)
#     return mm

def easy_for_folder(m):
    # decorator making a function act on folder instead of an array
    def mm(obj, folder, *args, **kwargs):
        images = get_images(folder=folder)
        return easy_for_images(m)(obj, images, *args, **kwargs)
    return mm


def easy_for_images(m):
    # decorator making a function act on folder instead of an array
    def mm(obj, images, *args, **kwargs):
        image = images[0]
        size = image.size
        mode = image.mode
        X = images2matrix(images, size=size)
        if obj.size is None:
            obj.size = size
        if obj.mode is None:
            obj.mode = mode
        return m(obj, X, *args, **kwargs)
    return mm


def easy_for_arrays(m):
    def mm(obj, arrays, *args, **kwargs):
        shape = arrays[0].shape
        obj.size = shape[:2]
        if len(shape) == 3:
            obj.n_channels = shape[2]
        else:
            obj.n_channels = 1
        X = np.array([a.ravel() for a in arrays])
        if obj.size is None:
            obj.size = size
        if obj.mode is None:
            if obj.n_channels == 1:
                obj.mode = 'L'
            elif obj.n_channels == 3:
                obj.mode = 'RGB'
            elif obj.n_channels == 4:
                obj.mode = 'RGBA'
        return m(obj, X, *args, **kwargs)
    return mm


def expit(x):
    return np.uint8(np.round(256 * ss.expit(x) - 0.5))


def logit(x):
    return ss.logit((x+0.5) / 256)


class _BaseEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, model):
        """
        Keyword Arguments:
            model {Transformer} -- a transformer or encoder
        """
        self.model = model

    @property
    def eigens_(self):
        if hasattr(self.model, 'components_'):
            return self.model.components_
        else:
            return self.model.eigenvectors_

    def _fit(self, X):
        """fit method
        
        Arguments:
            X {np.ndarray} -- matrix of images, each row vector represent an image.
        """
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def inverse_transform(self, Y):
        return self.model.inverse_transform(Y)

    def fit(self, X, *args, **kwargs):
        X = np.asarray(X, dtype='float64')
        self._Xtrain = X
        return self._fit(X, *args, **kwargs)


class ImageEncoder(_BaseEncoder):
    __sigular_values = None

    def __init__(self, model, size=None, mode=None, n_channels=1):
        """
        Keyword Arguments:
            model {Transformer} -- a transformer such as PCA, ICA
            size {tuple} -- size of image (default: {None})
            mode {str} -- mode of image (default: {None})
        """
        super().__init__(model)
        self.size = size
        self.mode = mode
        self.n_channels = n_channels

    @property
    def coordinates_(self):
        return self.transform(self._Xtrain)

    @easy_for_folder
    def eztransform(self, X):
        return self.transform(X)

    @easy_for_folder
    def ezfit_transform(self, X):
        return self.fit_transform(X)

    @easy_for_arrays
    def array_transform(self, X):
        return self.transform(X)

    @property
    def eigen_images(self):
        """Get eigen images

        transform the eigens_ to images
        """
        def _minmaxmap(X, mi=0, ma=255):
            lb, ub = X.min(), X.max()
            return mi + (X - lb) / (ub - lb) * ma
        if self.n_channels == 1:
            size = self.size
        else:
            size = self.size + (self.n_channels,)
        return [vector2image(_minmaxmap(y), size, mode=self.mode) for y in self.eigens_]

    def inverse_transform(self, Y, toimage=False):
        # transform to images if toimage=True
        X = super().inverse_transform(Y)
        if toimage:
            if self.size is None:
                raise Exception("Attribute `size` (as the size of the images) should not be None!")
            return [vector2image(x, size=self.size, mode=self.mode) for x in X]
        else:
            return X

    def reconstruct(self, X, toimage=True):
        # reconstruction of images under KL basis
        return self.inverse_transform(self.transform(X), toimage=toimage)

    def error(self, X):
        # error of reconstruction
        R = self.reconstruct(X, toimage=False)
        X = images2matrix(X, True, size=self.size)
        return LA.norm(X-R, axis=1)

    @easy_for_images
    def fit(self, *args, **kwargs):
        # images have the same size and mode
        return super().fit(*args, **kwargs)

    @easy_for_folder
    def ezfit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    @easy_for_arrays
    def array_fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def dist(self, X):
        from scipy.spatial.distance import cdist
        return cdist(self.coordinates_, self.transform(X))

    def classify(self, X):
        D = self.dist(X)
        return D.argmin(axis=0)

    # def locate(image):
    #     pass

    def generate(self, size=1, toimage=False, standard=True, scale=None):
        from scipy.stats import gaussian_kde, norm
        from sklearn.decomposition import FastICA, NMF

        cc = self.coordinates_.T
        if isinstance(self.model, FastICA):
            _gen = lambda data, size: gaussian_kde(data).resample(size)
            c = np.row_stack([_gen(c, size) for c in cc])
            return self.inverse_transform(c.T, toimage=toimage)
        if isinstance(self.model, NMF):
            standard = False
        if standard:
            if scale is None:
                scale = self.model.singular_values_
            c = norm.rvs(scale=scale, size=(size, self.model.n_components_))
            return self.inverse_transform(c, toimage=toimage)
        else:
            _gen = lambda data, size: norm.rvs(*norm.fit(data), size=size)
            c = np.row_stack([_gen(c, size) for c in cc])
            return self.inverse_transform(c.T, toimage=toimage)

    def generate_grid(self, r=4, c=4, toimage=True, *args, **kwargs):
        if self.size is None:
            raise Exception("Attribute `size` (as the size of the images) should not be None!")
        h, w = self.size
        X = self.generate(r * c, toimage=False, *args, **kwargs)
        if self.n_channels == 3:
            X = np.vstack([np.hstack([X[i*c+j].reshape((h, w, 3)) for j in range(c)]) for i in range(r)])
        else:
            X = np.block([[[X[i*c+j].reshape((h, w))] for j in range(c)] for i in range(r)])
        X = np.insert(X, np.arange(1, r)*h, 0, axis=0)
        X = np.insert(X, np.arange(1, c)*w, 0, axis=1)

        if toimage:
            return Image.fromarray(X.astype('uint8')).convert(self.mode)
        else:
            return X
        return X


class LogitImageEncoder(ImageEncoder):
    def _fit(self, X):
        """fit method
        
        Arguments:
            X {np.ndarray} -- matrix of images, each row vector represent an image.
        """
        X = logit(X)
        return super()._fit(X)

    def transform(self, X, *args, **kwargs):
        X = logit(X)
        return super().transform(X, *args, **kwargs)

    def inverse_transform(self, Y, toimage=False):
        X = _BaseEncoder.inverse_transform(self, Y)
        X = expit(X)
        if toimage:
            if self.size is None:
                raise Exception("Attribute `size` (as the size of the images) should not be None!")
            return [vector2image(x, size=self.size, mode=self.mode) for x in X]
        else:
            return X


if __name__ == '__main__':
    from sklearn.decomposition import *
    
    def demo_face(folder=pathlib.Path.cwd(), *args, **kwargs):
        # save images in the folder before demo
        
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        # define a model, such as PCA, NMF
        model = PCA(*args, **kwargs)
        ip = ImageEncoder(model)
        # a user-friendly API calling `fit` method of the model
        ip.ezfit(folder=folder)
        # save the eigen images in `eigen/` subfolder
        save_folder = folder / 'eigen'
        save_folder.mkdir(exist_ok=True)
        for k, im in enumerate(ip.eigen_images):
            im.save(save_folder / f'{k}.jpg')
        # generate artificial images
        for k, im in enumerate(ip.generate(10, toimage=True)):
            im.save(save_folder / f'artificial{k}.jpg')

    def demo_digit(folder=pathlib.Path.cwd(), *args, **kwargs):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        from sklearn import datasets
        model = FastICA(*args, **kwargs)
        ip = ImageEncoder(model, size=(8,8))
        digists = datasets.load_digits()
        ip._fit(digists.data * (255//16))

        save_folder = folder / 'eigen'
        save_folder.mkdir(exist_ok=True)
        for k, im in enumerate(ip.eigen_images):
            im.save(save_folder / f'{k}.jpg')
        # generate artificial images
        for k, im in enumerate(ip.generate(10, toimage=True)):
            im.save(save_folder / f'artificial{k}.jpg')


    demo_digit(n_components=15)
    