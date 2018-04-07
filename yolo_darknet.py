import functools
from functools import partial, reduce

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

# Partial wrapper for Conv2D with same padding always
_darknet_conv2d = partial(Conv2D, padding='same')


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

        Reference: https://mathieularose.com/function-composition-in-python/
        """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence is not supported")


@functools.wraps(Conv2D)
def darknet_conv2d(*args, **kwargs):
    '''Wrapper to set Darknet weight regularizer for Conv2D.'''
    darknet_conv_kwargs = {'Kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _darknet_conv2d(*args, **kwargs)


def darknet_conv2d_bn_leaky(*args, **kwargs):
    '''Darknet Conv2D followed by batch normalization and leaky relu'''
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(darknet_conv2d(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    '''
    Bottleneck block of 3x3, 1x1, 3x3 convolutions.
    :param outer_filters:
    :param bottleneck_filters:
    :return:
    '''
    return compose(darknet_conv2d_bn_leaky(outer_filters, (3, 3)),
                   darknet_conv2d_bn_leaky(bottleneck_filters, (1, 1)),
                   darknet_conv2d_bn_leaky(outer_filters,(3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filter):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(bottleneck_block(outer_filters, bottleneck_filter),
                   darknet_conv2d_bn_leaky(bottleneck_filter, (1, 1)),
                   darknet_conv2d_bn_leaky(outer_filters,(3, 3)))


def darknet_body():
    """Generate first 18 conv layers of Darknet."""
    return compose(darknet_conv2d_bn_leaky(32, (3, 3)),
                   MaxPooling2D(),
                   darknet_conv2d_bn_leaky(64, (3, 3)),
                   MaxPooling2D(),
                   bottleneck_block(128, 64),
                   MaxPooling2D(),
                   bottleneck_block(256, 128),
                   MaxPooling2D(),
                   bottleneck_x2_block(512, 256),
                   MaxPooling2D(),
                   bottleneck_x2_block(1024, 512))


def darknet(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet_body()(inputs)
    logits = darknet_conv2d(1000, (1, 1), activation='softmax')(body)
    return Model(inputs, logits)
