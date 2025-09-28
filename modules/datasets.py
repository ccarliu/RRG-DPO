import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0].split(".")[0] + "-512" + ".jpg")).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset_DPO(BaseDataset):

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        

        loaded_data = np.load(args.retrieved_id)
        self.filted_name = loaded_data['filted_name']

        #self.filted_name = [l[1:] for l in self.filted_name]

        self.indices = loaded_data['indices']
        self.subject_indices = {k:v for k,v in zip(self.filted_name, self.indices)}

        annotation = json.load(open(os.path.join(args.ann_path),'r'))
        ann = annotation['train']

        # Sentence label
        with open(args.setence_label, 'rb') as file:
            sentences_label = pickle.load(file)

        self.subject_label = {}
        self.subject_sentence_label = {}
        nann = []
        self.removed_ids = []
        # self.subject_report = {}
        for iidx, l in enumerate(ann):
            cpath = l['image_path'][0].split("/")[2]
            if cpath in self.subject_indices.keys():
                nann.append(l)
                # nsentence_label.append(self.sentences_label[ii])
                self.subject_label["s" + str(l["study_id"])] = l["labels"]
                self.subject_sentence_label["s" + str(l["study_id"])] = sentences_label[iidx]
            else:
                self.removed_ids.append(iidx)
        ann = nann
        self.examples = ann
        print(len(self.examples), len(self.removed_ids))

        self.sname_to_idx = {}
        for idx, l in enumerate(self.examples):
            cpath = l['image_path'][0].split("/")[2]
            # print(cpath)
            self.sname_to_idx[cpath] = idx

        # further filter:
        # 读取保存的数据
        with open('results/mimic_cxr/model_best_prob.pkl', 'rb') as f:
            self.prob_results = pickle.load(f)

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

        self.ori_example = []
        nannot = []
        for l in self.examples:
            cscore = self.prob_results[l['id']]
            self.ori_example.append(l)
            if cscore[0]>-1.2 and cscore[1]>-1.2:
                nannot.append(l)

        
        
        #self.examples = nannot
        print(len(self.examples))


    def get_label_similarity(self, vector1, vector2):
        
        # print(vector1, vector2)
        intersection = torch.sum(((vector1 == 1) & (vector2 == 1)).float()).item()
        
        union = torch.sum(((vector1 == 1) | (vector2 == 1))).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def _ids_map(self, cidx):
        tosub = 0
        for l in self.removed_ids:
            if cidx == l:
                return -1
            else:
                if cidx > l:
                    tosub += 1
        return cidx - tosub

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        caption = example['report']
        image = Image.open(os.path.join(self.image_dir, image_path[0].split(".")[0] + "-512" + ".jpg")).convert('RGB')

        #########################################################################
        csubject = image_path[0].split("/")[2]
        smin = 1
        idxmin = self.subject_indices[csubject][1]
        clabel = self.subject_label[csubject]

        acc_sentence_label = self.subject_sentence_label[csubject]

        if True:
            
            for iidx, l in enumerate(self.subject_indices[csubject][:20]):
                if idx == 0:
                    continue
                cs = self.get_label_similarity(torch.tensor(clabel), torch.tensor(self.subject_label[self.filted_name[l]]))
                if cs < smin:
                    smin = cs
                    idxmin = l

            rej_sentence_label = self.subject_sentence_label[self.filted_name[idxmin]]
            rej_label = self.subject_label[self.filted_name[idxmin]]
            # print(idxmin)
            # print(self.filted_name[idxmin])
            # print(self.sname_to_idx[self.filted_name[idxmin]])
            rej_cap = self.ori_example[self.sname_to_idx[self.filted_name[idxmin]]]["report"]
            rej_example = self.ori_example[self.sname_to_idx[self.filted_name[idxmin]]]
        else:
            for iidx, l in enumerate(self.examples[idx]["clip_indices"]):
                cidx = self._ids_map(l)
                if cidx == -1:
                    continue
                cs = self.get_label_similarity(torch.tensor(clabel), torch.tensor(self.examples[cidx]["labels"]))
                if cs < smin:
                    smin = cs
                    idxmin = cidx
            # print(smin)
            image_path = self.examples[idxmin]['image_path']
            csubject = image_path[0].split("/")[2]

            rej_sentence_label = self.subject_sentence_label[csubject]
            rej_label = self.examples[idxmin]["labels"]

            rej_cap = self.examples[idxmin]["report"]
            rej_example = self.ori_example[idxmin]



        ####################################sentence mask
        acc_sentence = caption.split(".")
        rej_sentence = rej_cap.split(".")

        attn_w = []
        attn_l = []

        for iidx in range(len(clabel)):
            w_ = clabel[iidx]
            l_ = rej_label[iidx]
            if iidx == 13:
                break
            if w_ != l_:
                if w_ == 1:
                    for ii, cl in enumerate(acc_sentence_label[1:]):
                        if cl[iidx] == 1:
                            attn_w.append(ii)
                            break
                    if l_ == 2:
                        for ii, cl in enumerate(rej_sentence_label[1:]):
                            if cl[iidx] == 2:
                                attn_l.append(ii)
                                break

                if l_ == 1:
                    for ii, cl in enumerate(rej_sentence_label[1:]):
                        if cl[iidx] == 1:
                            attn_l.append(ii)
                            break
                    if w_ == 2:
                        for ii, cl in enumerate(acc_sentence_label[1:]):
                            if cl[iidx] == 2:
                                attn_w.append(ii)
                                break
                    

        attn_w_idx = []
        attn_l_idx = []
        if len(attn_w) != 0:
            cstart = 1
            for iidx in range(len(acc_sentence_label[1:])):
                
                if iidx in attn_w:
                    attn_w_idx.append([cstart, cstart+len(acc_sentence[iidx].split())])
                cstart += len(acc_sentence[iidx].split()) + 1

        if len(attn_l) != 0:
            cstart = 1 # length of prompt
            for iidx in range(len(rej_sentence_label[1:])):
                
                if iidx in attn_l:
                    attn_l_idx.append([cstart, cstart+len(rej_sentence[iidx].split())])
                cstart += len(rej_sentence[iidx].split()) + 1

        # for place hold
        while len(attn_w_idx) < 10:
            attn_w_idx.append([0,0])
        while len(attn_l_idx) < 10:
            attn_l_idx.append([0,0])
        
        attn_w_idx = attn_w_idx[:10]
        attn_l_idx = attn_l_idx[:10]

        ####################################sentence mask

        #########################################################################

        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)


        
        report_masks_rej = rej_example['mask']
        report_ids_rej = rej_example['ids']
        seq_length_rej = len(report_ids_rej)

        ###############################################

        cap_mask_tensor = np.array(report_masks)
        cap_mask_rej_tensor = np.array(report_masks_rej)

        
        for idxs in attn_w_idx:
            if idxs[0] == 0:
                break
            cap_mask_tensor[idxs[0]:idxs[1]] += 4
        for idxs in attn_l_idx:
            if idxs[0] == 0:
                break
            cap_mask_rej_tensor[idxs[0]:idxs[1]] += 4
        

        ###############################################
        sample = (image_id, image, report_ids, report_masks, seq_length, report_ids_rej, report_masks_rej, seq_length_rej, cap_mask_tensor, cap_mask_rej_tensor)
        return sample
