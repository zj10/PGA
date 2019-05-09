from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
from functools import partial
import json
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
# from tensorflow_probability import distributions as tfd
import tflib as tl
import utils
import fid

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset_name', default='mnist', choices=['mnist', 'cifar10', 'celeba'])
parser.add_argument('--model', dest='model_name', default='conv_mnist', choices=['conv_mnist', 'conv_32', 'conv_64'])
parser.add_argument('--epoch', dest='epoch', type=int, default=600)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--lr', dest='lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--bn', dest='use_bn', type=lambda v: v.lower() in ('true', 'yes'), default=False,
                    help='use batchnorm or not')
parser.add_argument('--z_dim', dest='z_dim', type=int, default=12, help='dimension of latent space')
parser.add_argument('--init_steps', dest='init_steps', type=int, default=3000, help='initialization steps')
parser.add_argument('--zn_rec', dest='zn_rec_coeff', type=float, default=6e-2,
                    help='coefficient of latent reconstruction loss (z~N)')
parser.add_argument('--zh_rec', dest='zh_rec_coeff', type=float, default=0.,
                    help='coefficient of latent reconstruction loss (z~H)')
parser.add_argument('--nll', dest='nll_coeff', type=float, default=1.5e-2, help='coefficient of NLL loss')
parser.add_argument('--experiment_name', dest='experiment_name',
                    default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_name
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
use_bn = args.use_bn
z_dim = args.z_dim
init_steps = args.init_steps
zn_rec_coeff = args.zn_rec_coeff
zh_rec_coeff = args.zh_rec_coeff
nll_coeff = args.nll_coeff
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
inception_path = fid.check_or_download_inception('../data/inception_model/')
fid_stats_dict = {'mnist': '../data/fid/fid_stats_mnist_train.npz',
                  'cifar10': '../data/fid/fid_stats_cifar10_train.npz',
                  'celeba': '../data/fid/fid_stats_celeba.npz'}
fid_stats_path = fid_stats_dict[dataset_name] if dataset_name in fid_stats_dict else None

# dataset and models
Dataset, img_shape, get_imgs = utils.get_dataset(dataset_name)
dataset = Dataset(batch_size=batch_size)
# TODO: use a separate validation set
dataset_val = Dataset(batch_size=100)
Enc, Dec = utils.get_models(model_name)
Enc = partial(Enc, z_dim=z_dim, use_bn=use_bn)
Dec = partial(Dec, channels=img_shape[2], use_bn=use_bn)


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

def enc_dec(img, is_training=True):
    # encode
    z = Enc(img, is_training=is_training)

    # decode
    img_rec = Dec(z, is_training=is_training)

    return z, img_rec


def dec_enc(z, is_training=True):
    # decode
    img = Dec(z, is_training=is_training)

    # encode
    z_rec = Enc(img, is_training=is_training)

    return z_rec


# input
img = tf.placeholder(tf.float32, [None] + img_shape)
normal_dist = tfd.MultivariateNormalDiag(scale_diag=np.ones([z_dim], dtype=np.float32))

# encode & decode
z, img_rec = enc_dec(img)
zn_targ, zh_targ = normal_dist.sample(batch_size), tf.stop_gradient(z)
zn_rec, zh_rec = dec_enc(zn_targ), dec_enc(zh_targ)


# loss
def log_det_jacobian(z):
    z = tf.stop_gradient(z)
    epsilon = 1e-2 * z_dim ** 0.5
    delta = tf.random_normal([batch_size, z_dim])
    delta = delta / tf.sqrt(tf.reduce_sum(tf.square(delta), 1, keepdims=True)) * epsilon
    return z_dim / 2 * tf.log(tf.reduce_sum(tf.square(1 / epsilon * (dec_enc(z + delta) - dec_enc(z))), 1))


img_rec_loss = tf.losses.mean_squared_error(img, img_rec)
zn_rec_loss, zh_rec_loss = tf.losses.mean_squared_error(zn_targ, zn_rec), tf.losses.mean_squared_error(zh_targ, zh_rec)
nll_enc_loss = -tf.reduce_mean(normal_dist.log_prob(z)) / z_dim
nll_dec_loss = tf.reduce_mean(log_det_jacobian(z)) / z_dim

global_step = tf.get_variable('global_steps', [], initializer=tf.zeros_initializer, trainable=False)
zn_rec_coeff = tf.cond(global_step > init_steps, lambda: zn_rec_coeff, lambda: 0.)
nll_coeff = tf.cond(global_step > init_steps, lambda: nll_coeff, lambda: 0.)
enc_loss = img_rec_loss + zn_rec_coeff * zn_rec_loss + nll_coeff * nll_enc_loss
if zh_rec_coeff > 0:
    zh_rec_coeff = tf.cond(global_step > init_steps, lambda: zh_rec_coeff, lambda: 0.)
    enc_loss += zh_rec_coeff * zh_rec_loss
dec_loss = img_rec_loss + nll_coeff * nll_dec_loss

# otpim
enc_vars = []
dec_vars = []
for var in tf.trainable_variables():
    if var.name.startswith('Enc'):
        enc_vars.append(var)
    elif var.name.startswith('Dec'):
        dec_vars.append(var)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
enc_gvs = optimizer.compute_gradients(enc_loss, enc_vars)
dec_gvs = optimizer.compute_gradients(dec_loss, dec_vars)
train_op = optimizer.apply_gradients(enc_gvs + dec_gvs, global_step=global_step)

# summary
summary = tl.summary({img_rec_loss: 'img_rec_loss',
                      zn_rec_loss: 'zn_rec_loss', zh_rec_loss: 'zh_rec_loss',
                      nll_enc_loss: 'nll_enc_loss', nll_dec_loss: 'nll_dec_loss'})

# sample
# TODO: compute running averages for different input batches respectively
z_intp_sample, img_rec_sample = enc_dec(img, is_training=True)
img_sample = Dec(normal_dist.sample([100]), is_training=True)
fid_sample = Dec(normal_dist.sample([1000]), is_training=True)
if dataset_name == 'mnist':
    fid_sample = tf.image.grayscale_to_rgb(fid_sample)

z_intp_split, img_split = tf.split(z_intp_sample, 2), tf.split(img, 2)
img_intp_sample = [Dec((1 - i) * z_intp_split[0] + i * z_intp_split[1], is_training=True) for i in np.linspace(0, 1, 9)]
img_intp_sample = [img_split[0]] + img_intp_sample + [img_split[1]]
img_intp_sample = tf.concat(img_intp_sample, 2)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

if fid_stats_path:
    with np.load(fid_stats_path) as stats:
        mu_real, sigma_real = stats['mu'][:], stats['sigma'][:]
    fid.create_inception_graph(inception_path)

# train
try:
    img_ipt_sample = get_imgs(dataset_val.get_next())
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    it = -1
    for ep in range(epoch):
        dataset.reset()
        it_per_epoch = it_in_epoch if it != -1 else -1
        it_in_epoch = 0
        for batch in dataset:
            it += 1
            it_in_epoch += 1

            # batch data
            img_ipt = get_imgs(batch)

            # train & add summary
            if (it + 1) % 100 == 0:
                summary_opt, _ = sess.run([summary, train_op], feed_dict={img: img_ipt})
                summary_writer.add_summary(summary_opt, it)
            else:
                sess.run([train_op], feed_dict={img: img_ipt})

            # display
            if (it + 1) % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, it_in_epoch, it_per_epoch))

            # sample
            if (it + 1) % 2000 == 0:
                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)

                img_rec_opt_sample, img_intp_opt_sample = sess.run([img_rec_sample, img_intp_sample],
                                                                   feed_dict={img: img_ipt_sample})
                img_rec_opt_sample, img_intp_opt_sample = img_rec_opt_sample.squeeze(), img_intp_opt_sample.squeeze()
                # ipt_rec = np.concatenate((img_ipt_sample, img_rec_opt_sample), axis=2).squeeze()
                img_opt_sample = sess.run(img_sample).squeeze()

                # im.imwrite(im.immerge(ipt_rec, padding=img_shape[0] // 8),
                #            '%s/Epoch_(%d)_(%dof%d)_img_rec.png' % (save_dir, ep, it_in_epoch, it_per_epoch))
                im.imwrite(im.immerge(img_intp_opt_sample, n_col=1, padding=0),
                           '%s/Epoch_(%d)_(%dof%d)_img_intp.png' % (save_dir, ep, it_in_epoch, it_per_epoch))
                im.imwrite(im.immerge(img_opt_sample),
                           '%s/Epoch_(%d)_(%dof%d)_img_sample.png' % (save_dir, ep, it_in_epoch, it_per_epoch))

                if fid_stats_path:
                    try:
                        mu_gen, sigma_gen = fid.calculate_activation_statistics(im.im2uint(
                            np.concatenate([sess.run(fid_sample).squeeze() for _ in range(5)], 0)), sess,
                            batch_size=100)
                        fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                    except:
                        fid_value = -1.
                    fid_summary = tf.Summary()
                    fid_summary.value.add(tag='FID', simple_value=fid_value)
                    summary_writer.add_summary(fid_summary, it)
                    print("FID: %s" % fid_value)

        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('Model is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()
