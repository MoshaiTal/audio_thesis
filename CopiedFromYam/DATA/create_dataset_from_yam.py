import os
import numpy as np
from torch.utils.data import Dataset

class ReverbDataset(Dataset):

    def __init__(self,mics_num,dataset_name,data_part,data_base_path,data_type='melspec',warmup_stage=False):
        super().__init__()
        self.data_type=data_type
        self.mics_num=mics_num
        self.clean_folder_path=os.path.join(data_base_path,f'clean_{data_type}',data_part)
        self.reverb_folder_path=os.path.join(data_base_path,f'reverb_{data_type}',data_part)
        self.clean_files, self.reverb_files =  self.get_all_files_paths()
        if warmup_stage:
            self.clean_files, self.reverb_files= self.clean_files[:20], self.reverb_files[:20]

        # self.time_bins_res = 80
        self.time_bins_res = 256

        print(f'Dataset properties:\n\tDataSet Name: {dataset_name}\n\tData Type: {data_type}\n\tData Part:{data_part}\n\tData Base-Path:{data_base_path}\n\tNumber Of Samples: {len(self.clean_files)}')

    def split_spec0(self, arr, x):
        if arr.ndim==3:
            pad_width = ((0, 0), (0, self.time_bins_res - arr.shape[1]), (0, 0))
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0) #-0.57822084
            num_slices=x//self.time_bins_res
            res=x % self.time_bins_res
            last_slice = np.expand_dims(arr_padded, axis=0)[:, :, :,
                         num_slices * self.time_bins_res:(num_slices + 1) * (self.time_bins_res)]
            last_slice[:, :, :, res:] = 0
            if num_slices > 0:
                splits = np.concatenate(
                    [arr_padded[:,:,:num_slices*self.time_bins_res].reshape(arr.shape[0], arr_padded.shape[1],-1, self.time_bins_res).transpose([2,0,1,3]),
                     last_slice],
                    axis=0)
            else:
                splits =last_slice
        else:
            pad_width = ((0, self.time_bins_res - arr.shape[0]), (0, 0))
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
            num_slices = x // self.time_bins_res
            res=x % self.time_bins_res
            last_slice = np.expand_dims(arr_padded, axis=0)[:, :,
                         num_slices * self.time_bins_res:(num_slices + 1) * (self.time_bins_res)]
            last_slice[:, :, res:] = 0
            if num_slices > 0:
                splits = np.concatenate(
                    [arr_padded[ :, :num_slices * self.time_bins_res].reshape(arr_padded.shape[0],-1,self.time_bins_res).transpose([1, 0, 2]),
                     last_slice],axis=0)
            else:
                splits = last_slice

        return splits

    def split_spec(self, arr):
        if arr.ndim==3:
            pad_width = ((0, 0), (0, self.time_bins_res - arr.shape[1]), (0, self.time_bins_res-np.mod(arr.shape[-1],self.time_bins_res)))
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0) #-0.57822084

            splits = arr_padded.reshape(arr.shape[0], arr_padded.shape[1],-1, self.time_bins_res).transpose([2,0,1,3])

        else:
            pad_width = ((0, self.time_bins_res - arr.shape[0]),  (0, self.time_bins_res-np.mod(arr.shape[-1],self.time_bins_res)))
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
            splits = arr_padded.reshape(arr_padded.shape[0],-1,self.time_bins_res).transpose([1, 0, 2])

        return splits

    def reconstruct_from_splits(self, splits):

        if splits.ndim == 4:
            reshaped = splits.transpose([1, 2, 0, 3]).reshape(splits.shape[1], splits.shape[2], -1)
            reshaped = reshaped[:, :80, :]
            pad_width = ((0, 0), (0,0), (0,3000- reshaped.shape[-1]))
            reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084

        else:
            reshaped = splits.transpose([1, 0,2]).reshape(splits.shape[1], -1)
            reshaped = reshaped[:80, :]
            pad_width = ((0, 0), (0, 3000 - reshaped.shape[-1]))
            reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084


        return reconstructed

    def get_all_files_paths(self):
        clean_file_path_l=[]
        reverb_file_path_l=[]

        for folder in os.listdir(self.clean_folder_path):
            clean_subfolder_path=os.path.join(self.clean_folder_path,folder)
            for file_name  in os.listdir(clean_subfolder_path):
                if not file_name.endswith('.txt'):
                    clean_file_path_l.append(os.path.join(clean_subfolder_path,file_name))
                    reverb_file_path_l.append(os.path.join(self.reverb_folder_path, folder,file_name))
        return clean_file_path_l, reverb_file_path_l

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        file_path_clean,file_path_reverb = self.clean_files[index], self.reverb_files[index]
        actual_index=int(file_path_clean[file_path_reverb.find('='):file_path_reverb.rfind(']')-1])
        if self.data_type == "recs":
           a=5
        elif self.data_type == "melspec":
            final_freq_idx = 80
        elif self.data_type == "spec":
            final_freq_idx = 256

        reverb_ch_l=[]
        for i in range(1,self.mics_num+1):
            file_path_reverb_new=file_path_reverb[:file_path_reverb.find('[')]+f'ch{i}_'+file_path_reverb[file_path_reverb.find('['):]
            reverb_ch_l.append(np.load(file_path_reverb_new))

        reverb    = np.array(reverb_ch_l)
        clean_org = np.load(file_path_clean)

        mask      = np.ones_like(clean_org)
        mask[final_freq_idx:] = 0
        mask[:,actual_index:] = 0

        mask=self.split_spec(mask)
        reverb=self.split_spec(reverb)
        clean=self.split_spec(clean_org)
        last_real_layer_idx= actual_index // self.time_bins_res
        last_real_time_bin_idx= actual_index % self.time_bins_res

        return reverb, clean, last_real_layer_idx, last_real_time_bin_idx,mask,file_path_clean.split(os.sep)[-3:]




