import torch

from model import *
import  argparse
import json
import  sys
import  os
import  numpy as np
import  h5py
import csv
#import  tensorflow        as      tf
from    collections       import namedtuple
from    src.utils         import  *
#from    src.preprocessor_Gen  import  *
from    src.config_VA        import  *
from    sys               import platform
from    tqdm              import tqdm
import dataloader_Gen as dataloader
#from resnet_transformer import *
import torch.nn.functional as F
import sys
import json
import pickle
from    src.utils_nlp   import *
import nltk
from nltk.stem import WordNetLemmatizer



def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)  # 将句子分词为单词
    lemmas = [lemmatizer.lemmatize(token,pos='v') for token in tokens]  # 对每个单词进行词形还原
    return lemmas
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
savepath_dict = "cap2"
with open(os.path.join(config1.h5path + '{}/{}/idx_to_word.pkl'.format(savepath_dict, 'train')), 'rb') as f:
  idx_to_word = pickle.load(f)

with open(os.path.join(config1.h5path + '{}/{}/word_to_idx.pkl'.format(savepath_dict, 'train')), 'rb') as f:
  word_to_idx = pickle.load(f)

  # def beam_search(model, input, S, k=2, max_length=18):
  #   k_prev_words = torch.full((k, 1), S, dtype=torch.long)  # (k, 1)
  #   next_word_inds=torch.tensor([word_to_idx["the"],word_to_idx["the"],word_to_idx["the"],word_to_idx["the"]])
  #   k_prev_words = torch.cat([k_prev_words, next_word_inds.unsqueeze(1)], dim=1)
  #   next_word_inds = torch.tensor([word_to_idx["car"], word_to_idx["car"], word_to_idx["car"],word_to_idx["car"]])
  #   k_prev_words = torch.cat([k_prev_words, next_word_inds.unsqueeze(1)], dim=1)
  #   # 此时输出序列中只有sos token
  #   seqs = k_prev_words  # (k, 1)
  #   #print(k_prev_words.shape)
  #   # 初始化scores向量为0
  #   Top_k_scores = torch.ones(k, 1)
  #   complete_seqs = list()
  #   complete_seqs_scores = list()
  #   step = 1
  #   hidden = model.vit(input)  # h_0: (1, k, hidden_size)
  #   hidden=torch.stack([hidden,hidden,hidden,hidden])
  #   #print(hidden.shape)
  #
  #   while True:
  #     outputs = model.decoder(k_prev_words, hidden)  # outputs: (k, seq_len, vocab_size)
  #     #print(k_prev_words.shape)
  #     outputs=outputs.reshape(k,-1,1304)
  #     outputs=F.softmax(outputs,dim=-1)
  #     #print(outputs.shape)
  #     next_token_logits = outputs[:, -1, :]  # (k, vocab_size)
  #     #print(next_token_logits.shape)
  #     if step == 1:
  #       # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
  #       top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
  #     else:
  #       # 此时要先展开再计算topk，如上图所示。
  #       # top_k_scores: (k) top_k_words: (k)
  #       top_k_scores, top_k_words = next_token_logits.reshape(-1).topk(k, 0, True, True)
  #     prev_word_inds = top_k_words // 1304  # (k)  实际是beam_id
  #     #print(prev_word_inds)
  #     next_word_inds = top_k_words % 1304  # (k)  实际是token_id
  #     #print(next_word_inds)
  #     # seqs: (k, step) ==> (k, step+1)
  #     seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
  #     Top_k_scores = Top_k_scores[prev_word_inds] * top_k_scores.unsqueeze(1)
  #     #print(seqs)
  #
  #     # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
  #     incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
  #                        next_word != word_to_idx['E']]
  #     # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
  #     complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
  #
  #     if len(complete_inds) > 0:
  #       complete_seqs.extend(seqs[complete_inds].tolist())  # 加入句子
  #       complete_seqs_scores.extend(Top_k_scores[complete_inds])  # 加入句子对应的累加log_prob
  #     # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
  #     k -= len(complete_inds)
  #
  #     if k == 0:  # 完成
  #       break
  #
  #     # 更新下一次迭代数据, 仅专注于那些还没完成的句子
  #     seqs = seqs[incomplete_inds]
  #     hidden = hidden[prev_word_inds[incomplete_inds]]
  #     Top_k_scores = Top_k_scores[incomplete_inds]  # (s, 1) s < k
  #     k_prev_words = seqs  # (s, 1) s < k
  #
  #     if step > max_length:  # decode太长后，直接break掉
  #       break
  #     step += 1
  #   i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
  #   # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
  #   seq = complete_seqs[i]
  #
  #   return seq



  def beam_search(model, input, S,speed,course, k=2, max_length=18):
    k_prev_words = torch.full((k, 1), S, dtype=torch.long)  # (k, 1)

    # 此时输出序列中只有sos token
    seqs = k_prev_words  # (k, 1)
    #print(k_prev_words.shape)
    # 初始化scores向量为0
    top_k_scores = torch.zeros(k, 1)
    complete_seqs = list()
    complete_seqs_scores = list()
    step = 1
    hidden = model.vit(input)  # h_0: (1, k, hidden_size)
    hidden= torch.cat((hidden, speed), dim=1)
    hiddden = torch.cat((hidden, course), dim=1)

    hidden=torch.stack([hidden]*k)
    #print(hidden.shape)

    while True:
      outputs = model.decoder(k_prev_words, hidden)  # outputs: (k, seq_len, vocab_size)
      #print(k_prev_words.shape)
      outputs=outputs.reshape(k,-1,1304)
      outputs=F.softmax(outputs,dim=-1)
      #print(outputs.shape)
      next_token_logits = outputs[:, -1, :]  # (k, vocab_size)
      #print(next_token_logits.shape)
      if step == 1:
        # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
        top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
      else:
        # 此时要先展开再计算topk，如上图所示。
        # top_k_scores: (k) top_k_words: (k)
        top_k_scores, top_k_words = next_token_logits.reshape(-1).topk(k, 0, True, True)
      prev_word_inds = top_k_words // 1304  # (k)  实际是beam_id
      #print(prev_word_inds)
      next_word_inds = top_k_words % 1304  # (k)  实际是token_id
      #print(next_word_inds)
      # seqs: (k, step) ==> (k, step+1)
      seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
      #print(seqs)

      # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
      incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                         next_word != word_to_idx['E']]
      # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
      complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

      if len(complete_inds) > 0:
        complete_seqs.extend(seqs[complete_inds].tolist())  # 加入句子
        complete_seqs_scores.extend(top_k_scores[complete_inds])  # 加入句子对应的累加log_prob
      # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
      k -= len(complete_inds)

      if k == 0:  # 完成
        break

      # 更新下一次迭代数据, 仅专注于那些还没完成的句子
      seqs = seqs[incomplete_inds]
      hidden = hidden[prev_word_inds[incomplete_inds]]
      top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)  # (s, 1) s < k
      k_prev_words = seqs  # (s, 1) s < k

      if step > max_length:  # decode太长后，直接break掉
        break
      step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
    # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
    seq = complete_seqs[i]

    return seq


def greedy_decoder(model, x, start_symbol,speed,course,goal,accel):
  """贪心编码
  For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
  target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
  Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
  :param model: Transformer Model
  :param enc_input: The encoder input
  :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
  :return: The target input
  """
  new_shape = [1, 21]
  input = torch.zeros(new_shape,dtype=torch.int64)

  # speed = model.fc1(speed)

  course = model.fc2(course)
  goal = model.fc3(goal)
  acc = model.fc4(accel)
  # x = torch.cat((x, speed), dim=1)
  x = torch.cat((x, course), dim=1)
  x = torch.cat((x, goal), dim=1)
  x = torch.cat((x, acc), dim=1)
  x = model.encoder1(x)
  x = model.encoder2(x)
  # x = model.encoder3(x)
  # x = model.encoder4(x)
  # x = model.encoder5(x)

  #enc_outputs = model.swin(enc_input)
  #enc_outputs= enc_outputs.reshape(-1, 10, 196, 768)
  #enc_outputs = enc_outputs.reshape(-1, 1960, 768)


  # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
  dec_input = torch.zeros((1, 0),dtype=torch.int64)
  terminal = False
  next_symbol = start_symbol
  max_length = 20  # 设置最大生成长度
  length = 0  # 当前生成长度
  i=0
  while not terminal and length < max_length:
    # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
    dec_input = torch.cat([dec_input, torch.tensor([[next_symbol]], dtype=torch.int64)],
                          -1)

    #print(dec_input.shape)
    # input[0, :i+1] = dec_input
    # print(input)
    #print(dec_input.shape)
    # dec_input = dec_input.long()  # 将 dec_input 张量转换为 LongTensor 类型
    # enc_outputs = enc_outputs.long()  # 将 dec_inputs 张量转换为 LongTensor 类型
    dec_outputs = model.decoder1(dec_input, x)
    #dec_outputs=dec_outputs[:,i]



    #print(dec_outputs.shape)
    #print(dec_outputs)
    #print(dec_outputs.shape)
    #projected = model.projection(dec_outputs)
    prob = dec_outputs.max(dim=-1, keepdim=False)[1]
    #print(prob)
    # 增量更新（我们希望重复单词预测结果是一样的）
    # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
    # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
    next_word = prob.data[-1]
    i+=1
    next_symbol = next_word
    if next_symbol == word_to_idx["E"]:
      terminal = True
    # print(next_word)
    length += 1
  # greedy_dec_predict = torch.cat(
  #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
  #     -1)
  greedy_dec_predict = dec_input[:, 1:]
  #print(greedy_dec_predict)
  return greedy_dec_predict

def per_image_standardization(image):
  mean = np.mean(image)
  std = np.std(image)
  if std != 0:
    standardized = (image - mean) / std
  else:
    standardized = image - mean
  return standardized


def preprocess_input(inImg):
  # dst = np.reshape(inImg, (-1, config1.imgRow, config1.imgCol, config1.imgCh))
  outimg = np.reshape(inImg, (-1, 224, 224, 3))
  # outimg=np.zeros((dst.shape[0],224, 224, 3))
  # for i in range(dst.shape[0]):
  # outimg[i]=cv2.resize(dst[i],(224,224))
  # dst = np.reshape(inImg, (-1, config1.imgRow, config1.imgCol, config1.imgCh))
  outimg = np.array([per_image_standardization(frame) for frame in outimg])
  #outimg = np.reshape(outimg, (-1, 10, 224, 224, 3))
  # print(dst.shape)
  outimg = np.transpose(outimg, (0, 3, 1, 2))
  # print(dst.shape)
  # dst = np.reshape(dst, (config1.batch_size, -1,config1.imgRow, config1.imgCol))
  # print(dst.shape)
  # dst = torch.from_numpy(dst).float()  # 转换为torch.Tensor
  return outimg

def main(args):
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('--getscore',      type=bool,  default=False, help='get performance scores')
  #parser.add_argument('--showvideo',     type=bool,  default=False, help='show video')
  parser.add_argument('--useCPU',        type=bool,  default=False, help='without GPU processing')
  parser.add_argument('--validation',    type=bool,  default=False, help='use validation set')
  parser.add_argument('--gpu_fraction',  type=float, default=0.7,   help='GPU usage limit')
  parser.add_argument('--extractText',   type=bool,  default=True,  help='extract attention maps')
  args = parser.parse_args(args)
  print_captions = True
  np.set_printoptions(threshold=99999)


  timestamp = "20220513-212213"
  args.model = "./model/" + "/model_train23_1019_3.pth"
  args.savepath = "./result/LSTM_Gen1/"
  config1.timelen = 3500 + 3
  timelen = 3500
  config1.batch_size_gen = 1
  savepath_dict = "cap"


  if args.getscore:    check_and_make_folder(args.savepath)
  if args.extractText: check_and_make_folder(config1.h5path + "extracted_text7/")

  # Create VA model
  model =clip_decoder()
  #print(model)

  #logits_softmax = model.inference()
  checkpoint = torch.load(args.model)
  #print(checkpoint)
  model.load_state_dict(checkpoint)





  # Load the pretrained model

  if args.model is not None:

    print("\rLoaded the pretrained model: {}".format(args.model))



  data_val = dataloader.data_iterator(test=True, model="VA")  #validation=True

  seq_ls, cap_gt_ls, cap_log_ls, accel_gt_ls, accel_ls, c_gt_ls, c_ls = [],[],[],[],[],[],[]
  refs_desc, hypo_desc, refs_just, hypo_just = [], [], [], [] #refs are GT, hypo are logits
  ref_desc,ref_just,pr_desc,pr_just={},{},{},{}
  k=1

  for bv in range(1216): #Size of test set #TODO Change according to dataset.
    caption,  goal, speed, timestamp, _, course, accel, img= next(data_val)
    # speed = speed[:, :5]
    # # print(speed.shape)
    # goal = goal[:, :5]
    # accel = accel[:, :5]
    # course=course[:,:5]
    #print(caption)

    speed = torch.from_numpy(speed)
    course = torch.from_numpy(course)
    goal = torch.from_numpy(goal)
    accel = torch.from_numpy(accel)

    # print(speed.shape)
    # print(course.shape)

    # speed = speed.squeeze()
    goal=goal.unsqueeze(1)
    speed = speed.unsqueeze(1)
    accel = accel.unsqueeze(1)
    # print(speed.shape)
    # print(speed)
    speed = speed.float()
    course = course.float()
    goal=goal.float()
    accel = accel.float()

    # 将输入数组插值为[batch, 768, 1]的数组
    # speed = F.interpolate(speed, size=512, mode='linear')
    # accel = F.interpolate(accel, size=512, mode='linear')

    # course= course.squeeze()

    course = course.unsqueeze(1)
    # print(course.shape)

    # 将输入数组插值为[batch, 768, 1]的数组
    # course = F.interpolate(course, size=512, mode='linear')
    # goal = F.interpolate(goal, size=512, mode='linear')
    #print(accel)
    # Xprep_batch = preprocess_input(img)
    # Xprep_batch = torch.from_numpy(Xprep_batch)
    #Xprep_batch = Xprep_batch.to("cuda")
    Xprep_batch = img.float()

    # caption_onehot = (np.arange(config1.dict_size) == caption[..., None]).astype(int)
    # caption_onehot = np.squeeze(caption_onehot)
    caption = np.squeeze(caption)
    #print(caption.shape)
    model.eval()
    with torch.no_grad():
    # 将预处理后的数据输入到模型中进行推理

      #greedy_dec_predict =beam_search(model, Xprep_batch, word_to_idx["S"],speed,course,4,18)
      greedy_dec_predict =greedy_decoder(model, Xprep_batch, word_to_idx["S"],speed,course,goal,accel)
      print(greedy_dec_predict)
      caption_gt=[idx_to_word[t.item()] for t in caption]
      caption_pre=[idx_to_word[n.item()] for n in greedy_dec_predict[0]]
      print(caption_pre)
      #caption_pre = [idx_to_word[n] for n in greedy_dec_predict]
      for i in range(22):
        if caption_gt[i]=='<sep>':
          s1=i
        if caption_gt[i]=='E':
          e1=i
      for i in range(len(caption_pre)):
        if caption_pre[i]=='<sep>':
          s2=i
          break
      cap_text_desc=caption_pre[:s2]
      cap_text_desc = ' '.join(cap_text_desc)
      # cap_text_desc = lemmatize_sentence(cap_text_desc)
      # cap_text_desc = ' '.join(cap_text_desc)
      cap_text_just=caption_pre[s2+1:]

      cap_text_just = ' '.join(cap_text_just)
      # cap_text_just=lemmatize_sentence(cap_text_just)
      # cap_text_just = ' '.join(cap_text_just)
      cap_text_desc_gt=caption_gt[1:s1]

      cap_text_desc_gt = ' '.join(cap_text_desc_gt)
      # cap_text_desc_gt = lemmatize_sentence(cap_text_desc_gt)
      # cap_text_desc_gt = ' '.join(cap_text_desc_gt)
      print(cap_text_desc_gt)
      cap_text_just_gt=caption_gt[s1+1:e1]
      # cap_text_just_gt = ' '.join(cap_text_just_gt)
      # cap_text_just_gt = lemmatize_sentence(cap_text_just_gt)
      cap_text_just_gt = ' '.join(cap_text_just_gt)
      print(cap_text_just_gt)


      #print(greedy_dec_predict.squeeze())
      #print([idx_to_word[t.item()] for t in caption])
      ref_just["{}".format(k)]=[cap_text_just_gt]
      ref_desc["{}".format(k)]=[cap_text_desc_gt]
      pr_just["{}".format(k)] = [cap_text_just]
      pr_desc["{}".format(k)] = [cap_text_desc]
      k+=1
      print(k)

      #print([idx_to_word[n.item()] for n in greedy_dec_predict.squeeze()])
      #cap_logits = model(Xprep_batch)
      #print(cap_logits.shape)
      #cap_logits = cap_logits.item()



    #cap_logits = sess.run([logits_softmax], feed_dict)

  #   if print_captions == True:
  #     cap_text = convert_cap_vec_to_text(cap_logits[0], idx_to_word)
  #     print("\n")
  #     print(cap_text)
  #     cap_text_gt = convert_cap_vec_to_text_gc(caption, idx_to_word)
  #     print(cap_text_gt)
  #
  #   # Get indices
  #   start_desc, end_desc, start_just, end_just = get_rel_indices(cap_logits[0], word_to_idx)
  #
  #   # Convert idx to words
  #   if start_desc != -1: # check if sentence has a separator
  #     #print(cap_logits[start_desc:end_desc+1])
  #     cap_text_desc = convert_cap_vec_to_text(cap_logits[0][start_desc:end_desc+1], idx_to_word)
      print(cap_text_desc)
  #     cap_text_just = convert_cap_vec_to_text(cap_logits[0][start_just:end_just+1], idx_to_word)
      print(cap_text_just)
  #
  #     # Save sentences in lists
      hypo_desc.append(str(cap_text_desc))
      hypo_just.append(str(cap_text_just))
  #
  #     # Get indices GT
  #     start_desc_gt, end_desc_gt, start_just_gt, end_just_gt = get_rel_indices_gt(caption, word_to_idx)
  #     cap_text_desc_gt = convert_cap_vec_to_text_gc(caption[start_desc_gt:end_desc_gt + 1], idx_to_word)
  #     cap_text_just_gt = convert_cap_vec_to_text_gc(caption[start_just_gt:end_just_gt + 1], idx_to_word)
      #print(cap_text_desc_gt)
      #print(cap_text_just_gt)
  #
      refs_desc.append(str(cap_text_desc_gt))
      refs_just.append(str(cap_text_just_gt))
  #   else: # if no separator generated, still append something to align with generated images. Should decrease score.
  #     hypo_desc.append("NULL")
  #     hypo_just.append("NULL")
  #     refs_desc.append("NOT-VALID")
  #     refs_just.append("NOT-VALID")
  #
  #   # # Save Imgs with Attn
  #   # img_att_np = visualize_attnmap_2(alps[0, 9, :], imgs[0, 9, :, :, :]) #0, 9 #TODO Change for BDDx or SAX
  #   # img_att = Image.fromarray(img_att_np)
  #   # set_n = "test/"
  #   # img_att.save("/root/Workspace/explainable-deep-driving-master/data/processed_full/img_w_attn/" + str(set_n) + str(bv) + ".png") #TODO change SAX/ full...
  #   #
  #   # seq_ls.append(str(np.squeeze(seq_id_b)))
  #   # cap_gt_ls.append(str(np.squeeze(caption_onehot)))
  #   # cap_log_ls.append(str(np.squeeze(cap_logits)))
  #   # accel_gt_ls.append(str(np.squeeze(accel_b)))
  #   # accel_ls.append(str(np.squeeze(pred_accel)))
  #   # c_gt_ls.append(str(np.squeeze(course_b)))
  #   # c_ls.append(str(np.squeeze(pred_course)))
  #
  #
  # if args.extractText:
  #   with open(config1.h5path + "extracted_text/" + "output.csv", 'w', newline='') as out:
  #     csv.writer(out, delimiter=' ').writerows(zip(seq_ls, cap_gt_ls, cap_log_ls, accel_gt_ls, accel_ls, c_gt_ls, c_ls))
  #
  # Save Lists
      print("***************************************\n")
  refs_just = [refs_just]
  refs_desc = [refs_desc]
  with open(config1.h5path + "extracted_text7/" + 'refs_just.pkl', 'wb') as f:
    pickle.dump(refs_just, f)
  with open(config1.h5path + "extracted_text7/" + 'refs_desc.pkl', 'wb') as f:
    pickle.dump(refs_desc, f)
  with open(config1.h5path + "extracted_text7/" + 'hypo_desc.pkl', 'wb') as f:
    pickle.dump(hypo_desc, f)
  with open(config1.h5path + "extracted_text7/" + 'hypo_just.pkl', 'wb') as f:
    pickle.dump(hypo_just, f)

  with open(config1.h5path + "extracted_text7/" + 'refs_just.json', 'w') as f:
    json.dump(ref_just, f)
  with open(config1.h5path + "extracted_text7/" + 'refs_desc.json', 'w') as f:
    json.dump(ref_desc, f)
  with open(config1.h5path + "extracted_text7/" + 'hypo_desc.json', 'w') as f:
    json.dump(pr_desc, f)
  with open(config1.h5path + "extracted_text7/" + 'hypo_just.json', 'w') as f:
    json.dump(pr_just, f)
  #
  # # Total Result
  print(bcolors.HIGHL + 'Done' + bcolors.ENDC)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
