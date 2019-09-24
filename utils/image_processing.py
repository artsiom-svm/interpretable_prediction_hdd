import tensorflow as tf
import matplotlib.pyplot as plt
import io
from pylab import annotate
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image(c, x, labels):
    figure = plt.figure(figsize=(30,10))
    ax_coeff = plt.subplot(1, 2, 1, title='Contribution coefficient')
    plt.xlabel('T, time')
    plt.yticks([])
    im = plt.imshow(c.T, cmap='hot')
    ax_coeff.set_aspect(4)
    # add labels
    for j,label in enumerate(labels):
        annotate(label, xy=(0, j), xytext=(-9, j), fontsize=10)

    plt.colorbar(im,fraction=0.026, pad=0.04)

    ax_contrib = plt.subplot(1, 2, 2, title='Total contribution')
    plt.xlabel('T, time')
    plt.yticks([])
    im = plt.imshow(x.T, cmap='hot')
    ax_contrib.set_aspect(4)
    # add labels
    for j,label in enumerate(labels):
        annotate(label, xy=(0, j), xytext=(-9, j), fontsize=10)

    plt.colorbar(im,fraction=0.026, pad=0.04)
    return figure

def save_images(coeff, contrib, writer, names, labels):
    with writer.as_default():
        for i,(c,x, n) in enumerate(zip(coeff, contrib, names)):
            img = image(c,x, labels)
            tf.summary.image(n, plot_to_image(img), step=0)
