import os
import joblib
import matplotlib
import torch
from sklearn.metrics import roc_auc_score
from GTC_code.Model_ import Model_TEXT, Model_CNN, Model_GNN

matplotlib.use('Agg')
from data_utils import HNC_OS_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = r'dataset'
csv_dir = r'csv_data'
all_data_path = r'csv_data\all'

train_dataset = HNC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir, set_name=r'train',
all_data_path=all_data_path,
)
val_dataset = HNC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir, set_name=r'validation',
all_data_path=all_data_path)
test_dataset = HNC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir, set_name=r'test',
all_data_path=all_data_path
)

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

input_dim = 1302
out_num = 1
model_GNN = Model_GNN(input_dim, out_num).to(device)

model_GNN.load_state_dict(
torch.load(r'best_model_GNN',
weights_only=False))
model_CNN = Model_CNN(1, 128, (64, 224, 224)).to(device)

model_CNN.load_state_dict(
torch.load(r'best_model_CNN.pth',
weights_only=False))

model_Text = Model_TEXT(1).to(device)
model_Text.load_state_dict(
torch.load(r'best_model_text.pth',
weights_only=False))

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

train_cnn_probs = []
train_gnn_probs = []
train_text_probs = []
train_targets = []

model_CNN.to(device)
model_GNN.to(device)
model_Text.to(device)

model_CNN.eval()
model_GNN.eval()
model_Text.eval()

with torch.no_grad():
for data in train_loader:
    data = data.to(device)
    cnn_p, _ = model_CNN(data)
    gnn_p, _ = model_GNN(data)
    text_p, _ = model_Text(data)
    target = data.y.cpu()

    
    train_cnn_probs.append(cnn_p.cpu().numpy())
    train_gnn_probs.append(gnn_p.cpu().numpy())
    train_text_probs.append(text_p.cpu().numpy())
    train_targets.append(target.numpy())
train_cnn_probs = np.vstack(train_cnn_probs)
train_gnn_probs = np.vstack(train_gnn_probs)
train_text_probs = np.vstack(train_text_probs)
train_targets = np.hstack(train_targets)

test_cnn_probs = []
test_gnn_probs = []
test_text_probs = []
test_targets = []
patient_list = []

with torch.no_grad():
for data in test_loader:
    cnn_p, _ = model_CNN(data)
    gnn_p, _ = model_GNN(data)
    text_p, _ = model_Text(data)
    target = data.y.cpu()
    patient = data.patient_id[0]


    test_cnn_probs.append(cnn_p.cpu().numpy())
    test_gnn_probs.append(gnn_p.cpu().numpy())
    test_text_probs.append(text_p.cpu().numpy())
    test_targets.append(target.numpy())
    patient_list.append(patient)
test_cnn_probs = np.vstack(test_cnn_probs)
test_gnn_probs = np.vstack(test_gnn_probs)
test_text_probs = np.vstack(test_text_probs)
test_targets = np.hstack(test_targets)
patient_list = np.hstack(patient_list)

def average_fusion(cnn_probs, gnn_probs, text_probs):
return (cnn_probs + gnn_probs + text_probs) / 3

def weighted_fusion(cnn_probs, gnn_probs, text_probs, cnn_weight=0.33, gnn_weight=0.34, text_weight=0.33):
return cnn_weight * cnn_probs + gnn_weight * gnn_probs + text_probs * text_weight

lr = GaussianNB()

train_fusion_features = np.column_stack([train_cnn_probs, train_gnn_probs, train_text_probs])
test_fusion_features = np.column_stack([test_cnn_probs, test_gnn_probs, test_text_probs])

lr.fit(train_fusion_features, train_targets)

joblib.dump(lr, os.path.join(r'Best_model_pth\fusion_gnb_model.joblib'))

nb_train_pred = lr.predict(train_fusion_features)
nb_test_pred = lr.predict(test_fusion_features)

if train_cnn_probs.shape[1] == 1:
train_cnn_probs_auc = train_cnn_probs.flatten()
train_gnn_probs_auc = train_gnn_probs.flatten()
train_text_probs_auc = train_text_probs.flatten()

text
test_cnn_probs_auc = test_cnn_probs.flatten()
test_gnn_probs_auc = test_gnn_probs.flatten()
test_text_probs_auc = test_text_probs.flatten()

avg_train_probs = average_fusion(train_cnn_probs, train_gnn_probs, train_text_probs).flatten()
avg_test_probs = average_fusion(test_cnn_probs, test_gnn_probs, test_text_probs).flatten()

weighted_train_probs = weighted_fusion(train_cnn_probs, train_gnn_probs, train_text_probs).flatten()
weighted_test_probs = weighted_fusion(test_cnn_probs, test_gnn_probs, test_text_probs).flatten()

nb_train_probs = lr.predict_proba(train_fusion_features)
nb_test_probs = lr.predict_proba(test_fusion_features)
if nb_train_probs.shape[1] == 2:
    nb_train_probs = nb_train_probs[:, 1]
    nb_test_probs = nb_test_probs[:, 1]
else:
    nb_train_probs = nb_train_probs.flatten()
    nb_test_probs = nb_test_probs.flatten()
else:
train_cnn_probs_auc = train_cnn_probs[:, 1]
train_gnn_probs_auc = train_gnn_probs[:, 1]
train_text_probs_auc = train_text_probs[:, 1]
test_cnn_probs_auc = test_cnn_probs[:, 1]
test_gnn_probs_auc = test_gnn_probs[:, 1]
test_textprobs_auc = test_text_probs[:, 1]


avg_train_probs = average_fusion(train_cnn_probs, train_gnn_probs, train_text_probs)[:, 1]
avg_test_probs = average_fusion(test_cnn_probs, test_gnn_probs, test_text_probs)[:, 1]

weighted_train_probs = weighted_fusion(train_cnn_probs, train_gnn_probs, train_text_probs)[:, 1]
weighted_test_probs = weighted_fusion(test_cnn_probs, test_gnn_probs, test_text_probs)[:, 1]

nb_train_probs = lr.predict_proba(train_fusion_features)[:, 1]
nb_test_probs = lr.predict_proba(test_fusion_features)[:, 1]
print("=== Single Model Performance ===")
print(f"CNN Training Accuracy: {accuracy_score(train_targets, train_cnn_probs.argmax(axis=1)):.4f}")
print(f"CNN Test Accuracy: {accuracy_score(test_targets, test_cnn_probs.argmax(axis=1)):.4f}\n")
print(f"CNN Training AUC: {roc_auc_score(train_targets, train_cnn_probs_auc):.4f}")
print(f"CNN Test AUC: {roc_auc_score(test_targets, test_cnn_probs_auc):.4f}")
print('=' * 50)
print(f"GNN Training Accuracy: {accuracy_score(train_targets, train_gnn_probs.argmax(axis=1)):.4f}")
print(f"GNN Test Accuracy: {accuracy_score(test_targets, test_gnn_probs.argmax(axis=1)):.4f}\n")
print(f"GNN Training AUC: {roc_auc_score(train_targets, train_gnn_probs_auc):.4f}")
print(f"GNN Test AUC: {roc_auc_score(test_targets, test_gnn_probs_auc):.4f}")
print('=' * 50)
print(f"Text Training Accuracy: {accuracy_score(train_targets, train_text_probs.argmax(axis=1)):.4f}")
print(f"Text Test Accuracy: {accuracy_score(test_targets, test_text_probs.argmax(axis=1)):.4f}\n")
print(f"Text Training AUC: {roc_auc_score(train_targets, train_text_probs_auc):.4f}")
print(f"Text Test AUC: {roc_auc_score(test_targets, test_text_probs_auc):.4f}")

print("\n=== Fusion Model Performance ===")
print(f"Naive Bayes Fusion - Training Accuracy: {accuracy_score(train_targets, nb_train_pred):.4f}")
print(f"Naive Bayes Fusion - Test Accuracy: {accuracy_score(test_targets, nb_test_pred):.4f}")
print(f"Naive Bayes Fusion - Training AUC: {roc_auc_score(train_targets, nb_train_probs):.4f}")
print(f"Naive Bayes Fusion - Test AUC: {roc_auc_score(test_targets, nb_test_probs):.4f}")

print("\n=== Best Fusion Strategy Detailed Report ===")
print(classification_report(test_targets, nb_test_pred))