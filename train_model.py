# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import models
import utils
import vgg

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session() as sess:
    
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced = models.resnet(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    #Losses
    '''
    NOTICE:
    We difine new texture losses bassed on some No-Reference Image Quality Assessment algorithms, 
    and a color loss based on Automatic Colorization task.
    Full code will be released upon the acceptance of the paper.
    For now, we replace them with a simpler but effeictive texture loss below and dped color loss.'''
#-------------------
    #  1) texture loss
    enhanced_x, enhanced_y=utils.compute_gradient(enhanced)
    dslr_image_x, dslr_image_y=utils.compute_gradient(dslr_image)

    loss_texture=tf.abs(enhanced_x-dslr_image_x)+tf.abs(enhanced_y-dslr_image_y)

    # 2) color loss
    enhanced_blur = utils.blur(enhanced)
    dslr_blur = utils.blur(dslr_image)

    loss_color= tf.reduce_sum(tf.pow(enhanced_blur - dslr_blur, 2))/(2 * batch_size)
#-------------------
    # 3) content loss

    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    # loss_content = 2 * tf.abs(tf.subtract(enhanced_vgg[CONTENT_LAYER] ,dslr_vgg[CONTENT_LAYER]) )/ content_size 
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size


    # 4) total variation loss
    #4.1 reference to athalye2015neuralstyle
    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = utils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = utils._tensor_size(enhanced[:,:,1:,:])

    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # final loss
    cost = w_content * loss_content + w_texture * loss_texture+w_color * loss_color + w_tv * loss_tv

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # tf.summary.scalar('lossSum', cost)
    # tf.summary.scalar('lossPSNR', loss_psnr)
    
    # tf.summary.image('enhanced_image', enhanced)
    # # tf.summary.image('ldr_image', phone_image[:,:,:,9:12])
    # tf.summary.image('ldr_image', phone_image[:,:,:,0:3])
    # tf.summary.image('hdr_image', dslr_image)
        
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)

    # optimize parameters of image enhancement networks

    train_params = [v for v in tf.global_variables() if v.name.startswith("generator")]

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

    #--------------
    #Optimizer
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = learning_rate
    # steps_per_epoch = 5000
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            int(steps_per_epoch), 0.99, staircase=True)
     
    # optimizer= tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    #         epsilon=1e-8, use_locking=False)
    # train_op=optimizer.minimize(cost, global_step=global_step, var_list = train_params)

    saver = tf.train.Saver(var_list=train_params, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    print('Training network')

    train_loss = 0.0

    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()

    for i in range(num_train_iters):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([cost, train_step],
                                        feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
        train_loss += loss_temp / eval_step

        # if i % 50 == 0:
        #     summary = sess.run(merged, feed_dict={phone_: phone_images, dslr_: dslr_images})
        #     writer.add_summary(summary, i)

        if i % eval_step == 0:

            # test generator and discriminator CNNs

            test_losses_gen = np.zeros((1, 6))
            test_accuracy_disc = 0.0
            loss_ssim = 0.0

            for j in range(num_test_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                phone_images = test_data[be:en]
                dslr_images = test_answ[be:en]

                [enhanced_crops, losses] = sess.run([enhanced, \
                                [cost, loss_content, loss_texture, loss_color, loss_tv, loss_psnr]], \
                                feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})

                test_losses_gen += np.asarray(losses) / num_test_batches
                # test_accuracy_disc += accuracy_disc / num_test_batches

                loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                                    enhanced_crops * 255) / num_test_batches


            logs_gen = "step %d, %s | generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                  (i, phone, train_loss, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                   test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], loss_ssim)

            print(logs_gen)

            # save the results to log file

            logs = open('models/' + phone + '.txt', "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})
             # print(enhanced_crops)

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.png', before_after)
                idx += 1

            train_loss = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
