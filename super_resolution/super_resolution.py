# -*- coding: utf-8 -*-

from PIL import Image
from chainer import Variable, Chain
from chainer import cuda, serializers
import chainer.functions as F
import chainer.links as L
import numpy as np

import random
import time

random.seed(time.time())


def toYCbCr(image):
    image = image.convert(mode="YCbCr")
    data = np.asarray(image)
    return np.array([data[:,:,0], data[:,:,1], data[:,:,2]], dtype=np.float32)


def toImageYCbCr(ycbcr, size):
    tmp = []
    for y, cb, cr in zip(ycbcr[0].flatten(), ycbcr[1].flatten(), ycbcr[2].flatten()):
        tmp.append((y, cb, cr))
    tmp_np = np.array(tmp)
    image_data = tmp_np.reshape((size, size, 3))
    return Image.fromarray(np.uint8(image_data), mode="YCbCr")


class ImageLoader:
    def __init__(self, image_paths):
        self._images = image_paths

    @classmethod
    def lower(cls, image):
        w, h = image.size
        small_image = image.resize((w/2, h/2))
        lower_image = small_image.resize((w, h))
        return lower_image


    @classmethod
    def random_crop(cls, image, crop=64):
        w, h = image.size
        w_padding = random.randrange(w - crop + 1)
        h_padding = random.randrange(h - crop + 1)
        cropped = image.crop((w_padding, h_padding, w_padding + crop, h_padding + crop))
        return cropped


    @classmethod
    def toYCbCr(cls, image):
        image = image.convert(mode="YCbCr")
        data = np.asarray(image)
        return np.array([data[:,:,0], data[:,:,1], data[:,:,2]], dtype=np.float32)


    def load(self, crop=64, outcrop=50, count=10000, batch_size=100):
        loop_count = count / batch_size
        for i in range(loop_count):
            xs = []
            ys = []
            count = 0
            error_count = 0
            while count < batch_size:
                try:
                    image_path = random.choice(self._images)
                    image = Image.open(image_path)
                    if any([x < crop for x in image.size]):
                        raise Exception("too small.")
                    cr_orig = self.random_crop(image, crop)
                    cr_low = self.lower(cr_orig)
                    pad = crop - outcrop
                    cr_orig = cr_orig.crop((pad/2, pad/2, crop - pad/2, crop - pad/2))
                    cr_orig_ycbcr = self.toYCbCr(cr_orig)
                    cr_low_ycbcr = self.toYCbCr(cr_low)

                    ys.append(cr_orig_ycbcr[0:1,:,:])
                    xs.append(cr_low_ycbcr[0:1,:,:])

                    count += 1
                    error_count = 0
                except:
                    error_count += 1
                    if error_count > 100:
                        raise
            yield (np.array(xs), np.array(ys))


class SRCNN(Chain):
    def __init__(self, in_size):
        super(SRCNN, self).__init__(
            conv1 = L.Convolution2D(1,   32,  3, stride=1),
            conv2 = L.Convolution2D(32,  64,  3, stride=1),
            conv3 = L.Convolution2D(64,  128, 3, stride=1),
            conv4 = L.Deconvolution2D(128, 64, 3, stride=1),
            conv5 = L.Deconvolution2D(64,  32, 3, stride=1),
            conv6 = L.Deconvolution2D(32,  1,  3, stride=1),
        )
        self._in_size = in_size

    @property
    def in_size(self):
        return self._in_size

    @property
    def out_size(self):
        return self._in_size

    @property
    def offset(self):
        return (self.in_size - self.out_size) / 2

    def __call__(self, x_data, y_data=None, train=True):
        x = Variable(x_data, volatile=not train)
        if train:
            t = Variable(y_data, volatile=not train)
        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = self.conv6(h)
        if train:
            return F.mean_squared_error(h, t)
        else:
            return h


def super_resolution(image, block_size=16, rate=2.0, model_path="superres.npz"):
    w, h = image.size
    image = image.resize((int(w*rate), int(h*rate)))

    exec(superres.sr_model)
    model = SRCNN(in_size=block_size)
    serializers.load_npz(model_path, model)

    offset = model.offset
    output_size = model.out_size
    h_blocks = (h / output_size) + (0 if (h % output_size == 0) else 1)
    w_blocks = (w / output_size) + (0 if (w % output_size == 0) else 1)

    tmp_h = offset + h_blocks * output_size + offset
    tmp_w = offset + w_blocks * output_size + offset

    pad_h1 = offset
    pad_w1 = offset

    padded_image = Image.new('RGB', (tmp_w, tmp_h), (255, 255, 255))
    padded_image.paste(image, (pad_w1, pad_h1, pad_w1 + w, pad_h1 + h))

    new_image = Image.new('RGB', (tmp_w, tmp_h), (255, 255, 255))
    new_image.paste(image, (pad_w1, pad_h1, pad_w1 + w, pad_h1 + h))
    new_image = new_image.convert(mode="YCbCr")

    for h_i in range(0, tmp_h, output_size):
        for w_i in range(0, tmp_w, output_size):
            if h_i + block_size - 1 <= tmp_h and w_i + block_size <= tmp_w:
                crop_image = padded_image.crop((w_i, h_i, w_i + block_size, h_i + block_size))
                crop_image_ycbcr = toYCbCr(crop_image)
                predicted = model(np.array([crop_image_ycbcr[0:1,:,:]]), train=False)
                offset_crop_image = toYCbCr(crop_image.crop((offset, offset, offset + output_size, offset + output_size)))
                d = cuda.to_cpu(predicted.data)[0]
                d[d[:,:] > 254] = 254
                d[d[:,:] < 1] = 1
                offset_crop_image[0] = d
                predicted_image = toImageYCbCr(offset_crop_image, output_size)
                new_image.paste(predicted_image, (w_i + offset, h_i + offset))
    new_image = new_image.crop((offset, offset, offset+w, offset+h))
    new_image = new_image.convert(mode="RGB")
    return image, new_image


def train(image_paths, epoch=100, epoch_size=10000, batch_size=100, block_size=16, gpu=0, model_path="superres.npz"):
    model = SRCNN(in_size=block_size)
    if gpu >= 0:
        cuda.init(gpu)
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    loader = ImageLoader(image_paths)

    for i in range(epoch):
        for xs, ys in loader.load(count=epoch_size, batch_size=batch_size, crop=block_size, outcrop=model.out_size):
            if gpu >= 0:
                xs = cuda.to_gpu(xs)
                ys = cuda.to_gpu(ys)
            model.zerograds()
            loss = model(xs, ys, train=True)
            loss.backward()
            optimizer.update()
    model.to_cpu()
    serializers.save_npz(model_path, model)
