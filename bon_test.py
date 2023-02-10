import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(tf.__version__,tf.keras.__version__) # 2.10.0 / 2.10.0
# print(device_lib.list_local_devices()) # gtx 1660 / com capa 7.5 
from keras.models import Model
from keras.layers import Lambda,GaussianNoise,Activation,Dense,Input
from keras.applications import VGG16,ResNet50V2
import keras.applications.resnet_v2 as resnet
import keras.applications.vgg16 as vgg
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score

#### Config Values ###

IMG_SHAPE = (224,224,3)
BATCH_SIZE = 32
EPOCHS = 40
SHUFFLE = True
SEED = 42
CLASSES =['cats']
TEST_CLS = ['raccoons','cats','dogs']
TRAIN_PATH = r"./data/train/"
TEST_PATH = r"./data/test/"

### utills 

def wrap_generator(generator):
    while True:
        x,y = next(generator)
        y = tf.keras.utils.to_categorical(y)
        zeros = tf.zeros_like(y) + tf.constant([1.,0.])
        y = tf.concat([y,zeros], axis=0)
        yield x,y

def get_label_test(test_gen):
    test_num = test_gen.samples
    label_test = []
    for i in range((test_num // test_gen.batch_size)+1):
        X,y = test_gen.next()
        label_test.append(y)
            
    label_test = np.argmax(np.vstack(label_test), axis=1)
    print(label_test.shape)
    
    return label_test

    #### Paper's Model ###

def get_model_paper(train=True):
        pre_process = Lambda(vgg.preprocess_input)
        vgg16 = VGG16(weights = 'imagenet', include_top = True, input_shape = (224,224,3))
        vgg16 = Model(vgg16.input, vgg16.layers[-3].output)
        vgg16.trainable = False
        
        inp = Input((224,224,3))
        vgg_16_process = pre_process(GaussianNoise(1e-8)(inp))
        vgg_out = vgg16(vgg_16_process)
        
        noise = Lambda(tf.zeros_like)(vgg_out)
        noise = GaussianNoise(0.01)(noise)

        if train:
            x = Lambda(lambda z: tf.concat(z, axis=0))([vgg_out,noise])
            x = Activation('relu')(x)
        else:
            x = vgg_out
            
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(2, activation='softmax')(x)

        model = Model(inp, out)
        model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), 
                    loss= tf.keras.losses.categorical_crossentropy,
                    # loss='binary_crossentropy',
                    metrics=['accuracy']
        )
        
        return model

### Define Dataset Generator

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
            TRAIN_PATH,
            target_size = (IMG_SHAPE[0], IMG_SHAPE[1]),
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            shuffle = SHUFFLE,
            seed=SEED,
            classes = CLASSES 
        )
test_generator = test_datagen.flow_from_directory(
            TEST_PATH,
            target_size = (IMG_SHAPE[0], IMG_SHAPE[1]),
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            shuffle = False,
            classes = CLASSES
        )


## biuld test
def wrap_generator_t(generator):
    while True:
        x,y = next(generator)
        y=y+1
        y = tf.keras.utils.to_categorical(y)
        zeros = tf.zeros_like(y) + tf.constant([1.,0.,0.])
        y = tf.concat([y,zeros], axis=0)
        yield x,y
    #     break
    # return x,y
# x,y = wrap_generator_t(test_generator)

### Train

train_model = get_model_paper()
with tf.device("/device:GPU:0"):
    train_model.fit(wrap_generator(train_generator),
                        steps_per_epoch=train_generator.samples//train_generator.batch_size, 
                        epochs=EPOCHS)


### Test

pred_model = get_model_paper(train=False)
pred_model.set_weights(train_model.get_weights())
ground_truth = get_label_test(test_generator)
pred_test = np.argmax(pred_model.predict(test_generator), axis=1)
main_acc = accuracy_score(ground_truth, pred_test)
print('ACCURACY:', main_acc)