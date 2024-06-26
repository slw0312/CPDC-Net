import  os
import  h5py
from    src.config_VA      import  *
from    src.utils_nlp   import *
from    src.utils       import *
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #-----------------------
    # Parameters
    #-----------------------

    param = dict2(**{
        "max_length":       20,    # the maximum length of sentences
        "vid_max_length":   1,    # the maximum length of video sequences
        "size_of_dict":     config1.dict_size, # the size of dictionary
        "chunksize":        4,    # for h5 format file writing, 10
        "savepath":         'cap4',
        "FINETUNING":       False,
        "SAVEPICKLE":       True })


    
    check_and_make_folder(config1.h5path+"cap4/log/")
    check_and_make_folder(config1.h5path+"cap4/feat/")

    #-----------------------
    # For each split, collect feats/logs
    #-----------------------
    #for split in ['train', 'test', 'val']:
    context_shapes = []
    cap_shapes = []
    data_samples = []
    for split in ['train', 'val', 'test']:
        check_and_make_folder(config1.h5path+param.savepath+'/'+split)

        # Step1: Preprocess caption data + refine captions
        caption_file = config1.h5path + 'captions_BDDX_' + split + '.json'
        #TODO change process_caption_data to process_caption_data_w_pos for PoS prediction
        annotations= process_caption_data(caption_file=caption_file,  max_length=param.max_length)
        if param.SAVEPICKLE: save_pickle(annotations, config1.h5path + '{}/{}/{}.annotations.pkl'.format(param.savepath, split, split))
        print(bcolors.BLUE   + '[main] Length of {} : {}'.format(split, len(annotations)) + bcolors.ENDC)

        # Step2: Build dictionary
        if param.FINETUNING:
            with open(os.path.join(config1.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                word_to_idx = pickle.load(f)
        else:
            if split == 'train':
                word_to_idx, idx_to_word = build_vocab(annotations=annotations, size_of_dict=param.size_of_dict)
                if param.SAVEPICKLE: save_pickle(word_to_idx, config1.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, split))
                if param.SAVEPICKLE: save_pickle(idx_to_word, config1.h5path + '{}/{}/idx_to_word.pkl'.format(param.savepath, split))
            else:
                with open(os.path.join(config1.h5path + '{}/{}/word_to_idx.pkl'.format(param.savepath, 'train')), 'rb') as f:
                    word_to_idx = pickle.load(f)

        # # Step3: Clustering
        # #if split == 'train': clusters, ind_cluster = cluster_annotations(annotations=annotations, k=20)
        #
        # # Step4: word to index
        # #  #TODO Diff Fct with POS or without
        captions= build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=param.max_length)
        # if param.SAVEPICKLE: save_pickle(captions, config1.h5path + '{}/{}/{}.captions.pkl'.format(param.savepath, split, split))

        # Step5: feat & masks #TODO change build_feat_matrix (BDDX) or build_feat_matrix_SAX (SAX)
        all_logs,  all_imgs4Cap, nr_samples_with_data = build_feat_matrix(
                                                           annotations=annotations,
                                                           max_length=param.vid_max_length,
                                                           fpath=config1.h5path,
                                                           FINETUNING=param.FINETUNING)

        # # Step6: Saving these data into hdf5 format
        feat = h5py.File(config1.h5path + "cap4/feat/" + split + ".h5", "w")
        logs = h5py.File(config1.h5path + "cap4/log/"  + split+ ".h5", "w")

        # dset = feat.create_dataset("/X",     data=all_feats4Cap, chunks=(param.chunksize, param.vid_max_length, 64, 12, 20) ) #fc8
        # dset = feat.create_dataset("/mask",  data=all_masks4Cap)
        dset = feat.create_dataset("/img",   data=all_imgs4Cap, chunks=(param.chunksize, param.vid_max_length, 384,640, 3))

        # dset = logs.create_dataset("/attn",  data=all_attns4Cap, chunks=(param.chunksize, param.vid_max_length, 240))
        # dset = logs.create_dataset("/context", data=all_contexts4cap, chunks=(param.chunksize, param.vid_max_length, 64))
        dset = logs.create_dataset("/Caption",      data=captions)

        dset = logs.create_dataset("/timestamp",    data=all_logs['timestamp'])
        dset = logs.create_dataset("/curvature",    data=all_logs['curvature'])
        dset = logs.create_dataset("/accelerator",  data=all_logs['accelerator'])
        dset = logs.create_dataset("/speed",        data=all_logs['speed'])
        dset = logs.create_dataset("/course",       data=all_logs['course'])
        dset = logs.create_dataset("/goaldir",      data=all_logs['goaldir'])
       # dset = logs.create_dataset("/pred_accel",   data=all_logs['pred_accel'])
        #23dset = logs.create_dataset("/pred_courses" ,data=all_logs['pred_courses'])

       # if split == 'train': dset = logs.create_dataset("/cluster",      data=ind_cluster)
       # print(all_contexts4cap.shape)
       #  print(captions.shape)
       # # print(all_logs['pred_accel'].shape)
       #  #context_shapes.append(all_contexts4cap.shape[0])
       #  cap_shapes.append(captions.shape[0])
       #  data_samples.append(nr_samples_with_data)
        print(bcolors.GREEN + '[main] Finish writing into hdf5 format: {}'.format(split) + bcolors.ENDC)

    #print(idx_to_word)
    #print(context_shapes)
    print(cap_shapes)
    print(data_samples)

if __name__ == "__main__":
    main()
