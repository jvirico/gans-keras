import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import tensorflow
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import image


##########################################
# Hyper-parameter section
##########################################
save_images_to_disk = True
## Image
img_rows = 32
img_cols = 32
channels = 3

## Latent space
z_dim = 100  #size of noise vector

## Training
iterations = 30000  #20000
batch_size = 128
sample_interval = 250
checkpoint_dir = "checkpoints/3"
ckp_every_X_iterations = 1000
##########################################
##########################################

os.makedirs("checkpoints/3", exist_ok=True)
os.makedirs("images/3", exist_ok=True)
os.makedirs("plots/3", exist_ok=True)


img_shape = (img_rows, img_cols, channels)
 


def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(128 * 16 * 16, input_dim=z_dim))
    model.add(LeakyReLU())
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(channels, kernel_size=7, padding='same', activation='tanh'))
    return model


def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=3, input_shape=img_shape))
    model.add(LeakyReLU())
    model.add(Conv2D(128, kernel_size=4))
    model.add(LeakyReLU())
    model.add(Conv2D(128, kernel_size=4))
    model.add(LeakyReLU())
    model.add(Conv2D(128, kernel_size=4))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    model = Sequential()
    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)
    return model


discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.00008, clipvalue=1.0, decay=1e-8),
                      metrics=['accuracy'])

generator = build_generator(z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8))


losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # Load the CIFAR10 dataset
    (X_train, y_train), (_, _) = cifar10.load_data()

    # Keep only one class
    X_train = X_train[y_train.flatten() == 3]
    #[0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck']

    

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #--  Train the Discriminator

        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        #--  Train the Generator

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        #Saving model checkpoints
        if((iteration+1) % iterations == 0 or ((iteration+1) < iterations and (iteration+1) % ckp_every_X_iterations == 0)):
            gan.save(checkpoint_dir + '/model3_' + str(iteration+1) + '.hdf5')


        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator, iteration)



image_grid_rows=5
image_grid_columns=13
fig = plt.figure(figsize=(image_grid_rows, image_grid_columns))
fig.show()
#reuse the same noise vector to visualise progression over time
z_sample_images = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))


def sample_images(generator, iteration):
    #z_sample_images = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))
    gen_imgs = generator.predict(z_sample_images)
    for i in range(gen_imgs.shape[0]):
        img = (gen_imgs[i, :, :, :] * 127.5 + 127.5)/255
        plt.subplot(image_grid_rows, image_grid_columns, i+1)
        plt.imshow(img)
        plt.axis('off')
    fig.canvas.draw()
    plt.savefig('image_at_{:04d}.png'.format(iteration))
    plt.pause(0.01)
    if(save_images_to_disk): plt.savefig('./images/3/it_'+str(iteration)+'.png')

    

train(iterations, batch_size, sample_interval)


stamp = 'model_52_it'+ str(iterations) +'_' + str(datetime.datetime.timestamp(datetime.datetime.now()))
losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
if(save_images_to_disk): plt.savefig('./plots/3/Losses_'+stamp+'.png')

accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
if(save_images_to_disk): plt.savefig('./plots/3/Accuracy_'+stamp+'.png')
plt.show()
