import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy, CategoricalAccuracy

from model import MyModel


# config 저장 디렉토리
config_filename = 'config.json'

# model 저장 디렉토리
saved_model_dir = Path('saved_model')
saved_model_filename = 'saved_model/model'

@tf.function
def train_step(X, y):
    with tf.GradientTape() as tape:
        predictions = model(X, training=True)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)

@tf.function
def val_step(X, y):
    predictions = model(X, training=False)
    t_loss = loss_object(y, predictions)

    val_loss(t_loss)
    val_accuracy(y, predictions)


if __name__ == "__main__":

    ########################################################################
    ## config
    ########################################################################

    # config 로딩
    print('config를 읽어옵니다...')

    with open(config_filename, 'rb') as f:
        config = json.load(f)   

    print('읽기 완료!\n')
    



    ########################################################################
    ## load data
    ########################################################################

    # train, val 데이터 로딩
    print('train, val 데이터를 읽어옵니다...')
    X_train = np.load('train_val_eval_data/X_train.npy')
    y_train = np.load('train_val_eval_data/y_train.npy')
    X_val = np.load('train_val_eval_data/X_val.npy')
    y_val = np.load('train_val_eval_data/y_val.npy')
    print('읽기 완료!\n')

    # train, val, eval 데이터셋으로 변환
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)


    # model 생성
    model = MyModel(config)

    # loss 함수와 옵티마이저 설정
    loss_object = CategoricalCrossentropy() # y label이 one-hot encoding
    optimizer = Adam()


    # train, val의 loss와 accuracy 설정
    train_loss = Mean(name='train_loss')
    train_accuracy = CategoricalAccuracy(name='train_accuracy')

    val_loss = Mean(name='val_loss')
    val_accuracy = CategoricalAccuracy(name='val_accuracy')



    ########################################################################
    ## train
    ########################################################################
    print('\n\n*** 학습을 시작합니다...')

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory='./tf_ckpts', 
                                        max_to_keep=config['epochs'])

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in range(config['epochs']):
        for X, y in tqdm(train_ds):
            train_step(X, y)

        for X, y in tqdm(val_ds):
            val_step(X, y)

        template = '학습 에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
        print (template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            val_loss.result(),
                            val_accuracy.result()*100))
        

        # save checkpoint
        save_path = manager.save()
        print(f'Saved checkpoint for epoch {int(ckpt.step)}: {save_path}\n')
        ckpt.step.assign_add(1)


    print('\n학습이 완료됐습니다!!!!\n')

    print('<Model 정보 요약>')
    model.summary()

    


    ########################################################################
    ## save the model
    ########################################################################

    # model이 저장될 디렉토리 확인 및 생성
    try:
        saved_model_dir.mkdir(parents=True, exist_ok=False)

    except FileExistsError:
        print()
        print(f'\n\nsaved_model directory: {saved_model_dir} is already there')

    # Save the entire model
    print('현재 상태의 모델을 저장합니다...')
    model.save(saved_model_filename)
    print('저장완료!')




    ########################################################################
    ## END
    ########################################################################
    print('\n\n\n~~~학습이 완료됐습니다~~~')

