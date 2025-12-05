import torch
from torch import nn
from torch_geometric.nn import TopKPooling, ResGatedGraphConv, global_max_pool, global_mean_pool
from torch.nn import functional as F
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, \
    ScaleIntensityd, Resized, RandFlipd
from transformers import AutoModel, AutoTokenizer


class Model_GNN(nn.Module):
    def __init__(self, input_dim, out_num, channel=64):
        super(Model_GNN, self).__init__()
        self.channel = channel
        self.out_num = out_num

        self.embed = nn.Linear(input_dim, 512)

        self.conv1 = self._build_conv_block(512, 256)
        self.pool1 = TopKPooling(256, ratio=0.9)

        self.conv2 = self._build_conv_block(256, channel * 2)
        self.pool2 = TopKPooling(channel * 2, ratio=0.8)

        self.conv3 = self._build_conv_block(channel * 2, channel * 4)
        self.pool3 = TopKPooling(channel * 4, ratio=0.7)

        self.concat_dim = 2 * (channel * 4)

        self.classifier = nn.Sequential(
            nn.Linear(self.concat_dim, self.channel * 4),
            nn.BatchNorm1d(self.channel * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.BatchNorm1d(self.channel * 2),
            nn.ReLU(),
            nn.Linear(self.channel * 2, self.out_num),
            nn.Sigmoid()
        )

    def _build_conv_block(self, in_channels, out_channels):
        return nn.ModuleList([
            ResGatedGraphConv(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            ResGatedGraphConv(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        ])

    def _apply_conv_block(self, block, x, edge_index):
        for layer in block:
            if isinstance(layer, ResGatedGraphConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def forward(self, data):
        data = data.cuda()

        x, edge_index, batch, = data.x, data.edge_index, data.batch,

        x = self.embed(x)
        x = F.relu(x)

        x = self._apply_conv_block(self.conv1, x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        feat1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self._apply_conv_block(self.conv2, x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        feat2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self._apply_conv_block(self.conv3, x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        feat3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # global_feat = torch.cat([feat1, feat2, feat3, ], dim=1)
        global_feat = feat3

        output = self.classifier(global_feat)
        return output, global_feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_kernel=2, pool_stride=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x)


class PubMedBertTextEncoder(nn.Module):
    def __init__(self,
                 pretrained_model_name=r"BiomedNLPPubMedBERT/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"):
        super(PubMedBertTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, texts):
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(next(self.bert.parameters()).device)
        attention_mask = encoding['attention_mask'].to(next(self.bert.parameters()).device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        return cls_embeddings


class Model_TEXT(nn.Module):
    def __init__(self, out_num=1):
        super(Model_TEXT, self).__init__()
        self.out_num = out_num
        self.text_encoder = PubMedBertTextEncoder()
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768 + 8, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, out_num),
            nn.Sigmoid()
        )

    def forward(self, data):
        texts = data.text
        patient_clins = data.patient_clin
        # encoder out
        encoder_out = self.text_encoder(texts)
        patient_clins = torch.tensor(patient_clins, dtype=torch.float32).cuda()
        encoder_out = torch.cat([patient_clins, encoder_out], dim=1)
        logits = self.classifier(encoder_out)
        return logits, encoder_out


class Model_CNN(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, input_size=(64, 224, 224), clin=None):
        super(Model_CNN, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.adaptive_pool = nn.AdaptiveAvgPool3d(self.input_size)
        self.channel_list = [32, 64, 128, 256]
        self.CT_conv_blocks = nn.Sequential(
            ConvBlock(self.in_channels * 2, self.channel_list[0] * 4, pool_kernel=(1, 2, 2), pool_stride=(1, 2, 2)),
            ConvBlock(self.channel_list[0] * 4, self.channel_list[1] * 2, pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2)),
            ConvBlock(self.channel_list[1] * 2, self.channel_list[2], pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2)),
            ConvBlock(self.channel_list[2], self.channel_list[3], pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2))
        )
        self._calculate_output_size(self.input_size[0], self.input_size[1], self.input_size[2])

        self.fc = nn.Linear(self.channel_list[-1] * self.output_size, latent_dim)

        self._initialize_weights()

        self.channel = 64

        if clin is not None:
            self.clin_channel = 9
        else:
            self.clin_channel = 0

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + self.clin_channel, self.channel * 4),
            nn.BatchNorm1d(self.channel * 4),
            nn.ReLU(),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.BatchNorm1d(self.channel * 2),
            nn.ReLU(),
            nn.Linear(self.channel * 2, 1),
            nn.Sigmoid()
        )

    def _load_and_preprocess_t(self, ct_paths_list, mask_paths_list, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                               target_size=(64, 224, 224)):
        import torch
        import numpy as np
        import SimpleITK as sitk

        batch_ct_tensors = []
        batch_mask_tensors = []

        
        for i, (ct_paths, mask_paths) in enumerate(zip(ct_paths_list, mask_paths_list)):
            try:
                
                ct_reader = sitk.ImageFileReader()
                ct_reader.SetFileName(ct_paths)  
                ct_image = ct_reader.Execute()

                
                ct_array = sitk.GetArrayFromImage(ct_image)

                
                mask_arrays = []
                for mask_path in mask_paths:
                    mask_reader = sitk.ImageFileReader()
                    mask_reader.SetFileName(mask_path)
                    mask_image = mask_reader.Execute()
                    mask_array = sitk.GetArrayFromImage(mask_image)
                    mask_arrays.append(mask_array)

                
                if len(mask_arrays) > 1:
                    merged_mask = np.max(np.stack(mask_arrays), axis=0)
                else:
                    merged_mask = mask_arrays[0]

                
                if ct_array.shape != merged_mask.shape:
                    print(f"warring: CT mask shape not same : CT={ct_array.shape}, mask={merged_mask.shape}")
                    from scipy.ndimage import zoom
                    zoom_factors = [
                        ct_array.shape[0] / merged_mask.shape[0],
                        ct_array.shape[1] / merged_mask.shape[1],
                        ct_array.shape[2] / merged_mask.shape[2]
                    ]
                    merged_mask = zoom(merged_mask, zoom_factors, order=0)  

               
                cropped_ct, cropped_mask = self._crop_around_mask_simple(ct_array, merged_mask, target_size)

                
                cropped_ct = np.clip(cropped_ct, a_min, a_max)
                cropped_ct = (cropped_ct - a_min) / (a_max - a_min) * (b_max - b_min) + b_min

                
                cropped_ct = (cropped_ct - cropped_ct.min()) / (cropped_ct.max() - cropped_ct.min() + 1e-8)

                
                ct_tensor = torch.from_numpy(cropped_ct).float().unsqueeze(0)  # shape: (1, depth, height, width)
                mask_tensor = torch.from_numpy(cropped_mask).float().unsqueeze(0)  # shape: (1, depth, height, width)

                batch_ct_tensors.append(ct_tensor)
                batch_mask_tensors.append(mask_tensor)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if batch_ct_tensors:
            batch_ct = torch.stack(batch_ct_tensors, dim=0)  # shape: (batch_size, 1, depth, height, width)
            batch_mask = torch.stack(batch_mask_tensors, dim=0)  # shape: (batch_size, 1, depth, height, width)
            return batch_ct, batch_mask
        else:
            return None, None

    def _crop_around_mask_simple(self, ct_array, mask_array, target_size):
        import numpy as np

        non_zero_coords = np.where(mask_array > 0)

        if len(non_zero_coords[0]) == 0:
            return self._center_crop_simple(ct_array, mask_array, target_size)

        center_z = int(np.mean(non_zero_coords[0]))
        center_y = int(np.mean(non_zero_coords[1]))
        center_x = int(np.mean(non_zero_coords[2]))

        depth, height, width = ct_array.shape

        target_depth, target_height, target_width = target_size

        start_z = max(0, center_z - target_depth // 2)
        end_z = min(depth, start_z + target_depth)

        start_y = max(0, center_y - target_height // 2)
        end_y = min(height, start_y + target_height)

        start_x = max(0, center_x - target_width // 2)
        end_x = min(width, start_x + target_width)

        if end_z - start_z < target_depth:
            if start_z == 0:
                end_z = min(depth, start_z + target_depth)
            else:
                start_z = max(0, end_z - target_depth)

        if end_y - start_y < target_height:
            if start_y == 0:
                end_y = min(height, start_y + target_height)
            else:
                start_y = max(0, end_y - target_height)

        if end_x - start_x < target_width:
            if start_x == 0:
                end_x = min(width, start_x + target_width)
            else:
                start_x = max(0, end_x - target_width)

        cropped_ct = ct_array[start_z:end_z, start_y:end_y, start_x:end_x]
        cropped_mask = mask_array[start_z:end_z, start_y:end_y, start_x:end_x]

        if cropped_ct.shape != target_size:
            cropped_ct = self._pad_to_size_simple(cropped_ct, target_size)
            cropped_mask = self._pad_to_size_simple(cropped_mask, target_size)

        return cropped_ct, cropped_mask

    def _center_crop_simple(self, ct_array, mask_array, target_size):
        depth, height, width = ct_array.shape
        target_depth, target_height, target_width = target_size

        start_z = max(0, (depth - target_depth) // 2)
        start_y = max(0, (height - target_height) // 2)
        start_x = max(0, (width - target_width) // 2)

        end_z = min(depth, start_z + target_depth)
        end_y = min(height, start_y + target_height)
        end_x = min(width, start_x + target_width)

        cropped_ct = ct_array[start_z:end_z, start_y:end_y, start_x:end_x]
        cropped_mask = mask_array[start_z:end_z, start_y:end_y, start_x:end_x]

        if cropped_ct.shape != target_size:
            cropped_ct = self._pad_to_size_simple(cropped_ct, target_size)
            cropped_mask = self._pad_to_size_simple(cropped_mask, target_size)

        return cropped_ct, cropped_mask

    def _pad_to_size_simple(self, image, target_size):

        import numpy as np

        current_size = image.shape
        pad_width = []

        for i in range(3):
            diff = target_size[i] - current_size[i]
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))

        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

        return padded_image

    def _calculate_output_size(self, d, h, w):
        self.output_size = d // 8 * h // 16 * w // 16

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):

        ct_paths, mask_paths = data.ct_path, data.all_mask_path
        ct_data, mask_data = self._load_and_preprocess_t(ct_paths, mask_paths)

        ct = torch.stack([torch.tensor(img) for img in ct_data])
        mask = torch.stack([torch.tensor(mask) for mask in mask_data])

        ct = self.adaptive_pool(ct.permute(0, 1, 4, 2, 3).contiguous())
        mask = self.adaptive_pool(mask.permute(0, 1, 4, 2, 3).contiguous())

        x = torch.cat([ct, mask], dim=1)

        ct_x = self.CT_conv_blocks(x.cuda())

        x = ct_x.view(ct_x.size(0), -1)

        z = self.fc(x)

        out = self.classifier(z)

        return out, z
