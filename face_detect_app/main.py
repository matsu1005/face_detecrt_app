import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import optimizers
import os
import numpy as np
import time

batch_size = 3
epochs = 10
IMG_HEIGHT = 100
IMG_WIDTH = 100


# トレーニングデータの作成
def create_train_data(path):
  image_gen_train = ImageDataGenerator(
                      rescale=1./255,
                      rotation_range=45,
                      width_shift_range=.15,
                      height_shift_range=.15,
                      horizontal_flip=True,
                      zoom_range=0.5
                      )

  train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                      directory=path,
                                                      shuffle=True,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode=None)
  return train_data_gen

# 検証データの作成
def create_val_data(path):
  image_gen_val = ImageDataGenerator(rescale=1./255)

  val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                  directory=path,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  class_mode=None)
  return val_data_gen


def create_model():
  input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
  # VGG16の図の緑色の部分（FC層）の作成
  top_model = Sequential()
  top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(3, activation='softmax'))

  # VGG16とFC層を結合してモデルを作成（完成図が上の図）
  vgg_model = Model(vgg16.input, top_model(vgg16.output))

  # VGG16の図の青色の部分は重みを固定（frozen）
  for layer in vgg_model.layers[:15]:
    layer.trainable = False

  # 多クラス分類を指定
  vgg_model.compile(loss='sparse_categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
  return vgg_model

def cul_data():
  history1 = vgg_model.fit_generator(
      train_data_gen,
      steps_per_epoch=3,
      epochs=30,
      validation_data=val_data_gen,
      validation_steps=1
  )