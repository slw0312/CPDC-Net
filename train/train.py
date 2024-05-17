
from model import *
import cv2
import os
import sys
import argparse
import json
import numpy as np
#from server import client_generator
#from src.LSTM_Gen_slim_v2_pos_penalties import * #TODO Change whether _pos, _w_penalties or pos_penalties or not
#from src.preprocessor_Gen import *
import torch
import torch.optim as optim
from config import *
from utils import *
import dataloader as dataloader
from PIL import Image
# from resnet_transformer import *
import datetime
#from src.preprocessor_CNN import *
import time
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def per_image_standardization(image):
    mean = np.mean(image)
    std = np.std(image)
    if std != 0:
        standardized = (image - mean) / std
    else:
        standardized = image - mean
    return standardized


def preprocess_input(inImg):
    dst = np.reshape(inImg, (-1, config1.imgRow, config1.imgCol, config1.imgCh))
    outimg=np.zeros((160,224, 224, 3))
    for i in range(160):
        outimg[i]=cv2.resize(dst[i],(224,224))
    #dst = np.reshape(inImg, (-1, config1.imgRow, config1.imgCol, config1.imgCh))
    outimg = np.array([per_image_standardization(frame) for frame in outimg])
    outimg = np.reshape(outimg, (-1, 224, 224, 3))
    #print(dst.shape)
    outimg = np.transpose(outimg, (0, 3, 1, 2))
    #print(dst.shape)
    #dst = np.reshape(dst, (config1.batch_size, -1,config1.imgRow, config1.imgCol))
    #print(dst.shape)
    # dst = torch.from_numpy(dst).float()  # 转换为torch.Tensor
    return outimg

def main():
    torch.cuda.init()
    #tf.random.set_random_seed(42777)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #pre_processor = PreProcessor_CNN_4frame()
    savepath_dict = "cap"

    #









    # with open(os.path.join(config1.h5path + '{}/{}/word_to_idx.pkl'.format(savepath_dict, 'train')), 'rb') as f:
    #     word_to_idx = pickle.load(f)
        #print(len(word_to_idx))

    #idx_sep = word_to_idx['<sep>']
    log_dir = config1.model_path_Gen + 'logs/' + current_time + "/"
    #model=ResNet_transform()
    model = clip_decoder().to(device)
    # 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                          momentum=0.99)  # 用adam的话效果不好
    #model=model.to("cpu")
    # Train over the dataset
    data_train = dataloader.data_iterator(model="VA")
    data_val = dataloader.data_iterator(validation=True, model="VA")



    val_loss = 9999999.0
    loss_sum = 0.0
    loss_val_sum = 0.0
    batch_counter = 0
    batch_counter_val = 0
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # decay_steps = config1.batches_per_dataset * 25
    decay_rate = 0.96
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    for i in range(config1.maxiter): #each epoch
        #i = i + 1
        loss_sum = 0.0
        loss_val_sum = 0.0
        start = time.time()

        for b in range(config1.batches_per_dataset): #TODO change if Batch size is changed

            #(captions_batch, masks_batch, goaldir_batch, speed_batch, timestamp_batch, seq_id_batch, course_batch,
           #  accel_batch, img_batch_s, pos_batch)
            caption,  _, _, timestamp, _, course, accel,img = next(data_train)
            Xprep_batch=preprocess_input(img)
            #Xprep_batch=torch.from_numpy(Xprep_batch)
            Xprep_batch=Xprep_batch.to("cuda")
            #Xprep_batch=Xprep_batch.float()



            # Convert caption to One-Hot-Encoded with dict size 10000

            #caption_onehot = (np.arange(config1.dict_size) == caption[..., None]).astype(int)  # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            #caption_onehot = np.squeeze(caption_onehot)
            caption=np.squeeze(caption)
            #print(caption.shape)
            #print(caption_onehot.shape)
            caption=torch.tensor(caption,dtype=torch.int64, device='cuda')
            Xpre=model(Xprep_batch,caption)
            optimizer.zero_grad()
            loss1=criterion(Xpre,caption)
            # 反向传播
            loss1.backward()

            # 更新参数
            optimizer.step()
            #lr_scheduler.step()
            #learning_rate_fn.step()

            # pos = pos - 1
            # pos_onehot = (np.arange(12) == pos[..., None]).astype(int)
            # pos_onehot = np.squeeze(pos_onehot)

            # Training
            # feed_dict = {Gen_model.context: context,
            #         Gen_model.pred_acc: pred_accel,
            #         Gen_model.pred_course: pred_course,
            #         Gen_model.features: feat,
            #         Gen_model.caption: caption,
            #         Gen_model.caption_onehot: caption_onehot,
            #         Gen_model.pos_gt: pos_onehot,
            #         Gen_model.idx_sep: idx_sep}
            #
            # _, l1loss, summary_train, _ = sess.run([train_op, loss, first_summary_train, first_summary_val], feed_dict)
            loss_sum += loss1
           # train_summary_writer.add_summary(summary_train, batch_counter)
            batch_counter += 1
          #


        # validation
        for bv in range(config1.batches_per_dataset_v): #TODO change if Batch size is changed
            with torch.no_grad():
                caption, masks, _, speed, timestamp, _, course, accel, img = next(data_val)
                Xprep_batch = preprocess_input(img)

                Xprep_batch = torch.from_numpy(Xprep_batch)
                Xprep_batch = Xprep_batch.float()
                Xprep_batch = Xprep_batch.to("cuda")
                # caption_onehot = (np.arange(config1.dict_size) == caption[..., None]).astype(
                #   #  int)  # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
                # caption_onehot = np.squeeze(caption_onehot)
                caption = np.squeeze(caption)
                caption = torch.tensor(caption,dtype=torch.int64, device='cuda')
                #print(caption.shape)
                Xpre = model(Xprep_batch,caption)
                optimizer.zero_grad()
                loss2 = criterion(caption, Xpre)
                loss_val_sum+=loss2



            #    img_p, _, acc_p, speed_p, course_p, _, goaldir_p, _ = pre_processor.process(sess, img, course, speed,
            #                                                                                curvature,



        #train_summary_writer.add_summary(summary_val, i)
        line = "Step {} | train loss: {} | val loss: {} ".format(i, loss_sum/config1.batches_per_dataset, loss_val_sum/config1.batches_per_dataset_v)
        print("%5.2f ms" % ((time.time() - start) * 1000.0))
        print("\rStep {} | train loss: {} | val loss: {} ".format(i, loss_sum/config1.batches_per_dataset, loss_val_sum/config1.batches_per_dataset_v))
        if os.path.exists("losses.txt"):
            with open("losses.txt", 'a') as f:
                # 追加内容到现有文件
                f.write(line)
                f.write('\n')
        else:
            with open("losses.txt", 'w') as f:
                f.write(line)
                f.write('\n')

        sys.stdout.flush()

        if val_loss > loss_val_sum:
            print("Last Val Loss: " + str(val_loss/config1.batches_per_dataset_v))
            val_loss = loss_val_sum
            #checkpoint_path = os.path.join(config.model_path_Gen, "model.ckpt")

            torch.save(model.state_dict(), 'model.pth')
            #filename = saver.save(sess, checkpoint_path)
           # print("Model saved in file: %s" % filename)
            with open( "losses.txt", 'a') as f:
                f.write("Model saved with val loss " + str(loss_val_sum/config1.batches_per_dataset_v))
                f.write('\n')


if __name__ == "__main__":
    main()
