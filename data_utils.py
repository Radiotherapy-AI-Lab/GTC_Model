import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class HNC_OS_Dataset(InMemoryDataset):
    def __init__(self, root, csv_root_path, set_name, all_data_path, transform=None,
                 pre_transform=None):
        self.csv_dir = csv_root_path
        self.name = set_name
        self.all_data_path = all_data_path
        self.clin_df = pd.read_csv(r'all_clin_data.csv')
        self.patient_clin_map = {}
        clinical_cols = [col for col in self.clin_df.columns if col != 'Patient_ID']
        for idx, row in self.clin_df.iterrows():
            self.patient_clin_map[row['Patient_ID']] = (row[clinical_cols])
        self.OS_data = r'train_data_csv'

        self.df_all_data = pd.read_csv(os.path.join(r'normalized_data_csv',
                                                    'normalized_data_' + self.name + '.csv'))

        self.case_text_mapping = {}
        self.csv_files_text = [
            r'Text_data\137_clinical_descriptions.csv',
            r'Text_data\298_clinical_descriptions.csv',
            r'Text_data\495_clinical_descriptions.csv',
            r'Text_data\606_clinical_descriptions.csv',
        ]
        self.data_frames = []

        for file in self.csv_files_text:
            try:
                df = pd.read_csv(file)
                self.data_frames.append(df)
            except Exception as e:
                print(f"read {os.path.basename(file)} error: {str(e)}")
        self.df_text_data = pd.concat(self.data_frames, ignore_index=True)
        for idx, row in self.df_text_data.iterrows():
            self.case_text_mapping[row['id']] = (row['description'])

        self.feature_list = self.df_all_data.columns[4:4 + 1302].tolist()
        self.feature_num = len(self.feature_list)

        self.paths_df = pd.read_csv(os.path.join(self.csv_dir, 'file_paths_CT.csv'))
        self.patient_path_map = {}
        for idx, row in self.paths_df.iterrows():
            self.patient_path_map[row['Patient_ID']] = (row['CT_Path'], row['Mask_Path'])

        self.df_label_data = pd.read_csv(os.path.join(self.OS_data, self.name + "_set.csv"))
        self.patient_label_map = {}
        for idx, row in self.df_label_data.iterrows():
            self.patient_label_map[row['Patient_ID']] = (row['label'])

        super(HNC_OS_Dataset, self).__init__(root, transform, pre_transform)
        if not os.path.exists(self.processed_paths[0]):
            1
        else:
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        save_dir = os.path.join(self.root, 'dataset_' + self.name, str(self.feature_num))
        os.makedirs(save_dir, exist_ok=True)
        return [os.path.join(save_dir, self.name + '_NPC.dataset')]

    def download(self):
        pass

    def process(self):
        grouped = self.df_all_data.groupby('Patient_ID')

        label_patient_ids = set(self.df_label_data['Patient_ID'].tolist())

        data_list = []
        for patient_id, group in tqdm(grouped):
            if patient_id not in label_patient_ids:
                continue
            component_encoder = LabelEncoder()
            sess_component_id = component_encoder.fit_transform(group.Component_ID)
            group = group.reset_index(drop=True)
            group['sess_component_id'] = sess_component_id

            length = len(group.Component_ID)
            node_features = group.loc[group.Patient_ID == patient_id, self.feature_list].values
            node_features = torch.tensor(node_features, dtype=torch.float)
            node_names = group['ROI_Name'].tolist()
            source_nodes = []
            target_nodes = []
            for i in range(length):
                for j in range(length):
                    if i != j:
                        source_nodes.append(i)
                        target_nodes.append(j)

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            text = self.case_text_mapping.get(patient_id, '')
            label_3 = self.patient_label_map.get(patient_id, None)
            all_mask_paths = group['Mask_Path'].tolist()
            y = torch.tensor([
                label_3,
            ], dtype=torch.float)

            patient_clin = self.patient_clin_map.get(patient_id, None)

            ct_path, mask_path = self.patient_path_map.get(patient_id, (None, None))
            if ct_path is None:
                continue
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                patient_id=patient_id,
                ct_path=ct_path,
                mask_path=mask_path,
                all_mask_path=all_mask_paths,
                text=text,
                patient_clin=patient_clin,
                node_names=node_names
            )
            data_list.append(data)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)

        self.data, self.slices = self.collate(data_list)
