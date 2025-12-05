import logging
import os
import time
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

log_dir = r'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"device: {device}")

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        pred, _ = model(data)
        label = data.y.view(-1, 1)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    return total_loss / total_samples


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true, y_score = [], []

    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data.to(device)
            pred, _ = model(data)
            label = data.y.view(-1, 1)

            loss = criterion(pred, label)
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

            y_true.append(label.cpu().numpy())
            y_score.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    return total_loss / total_samples


def main():

    dataset_path = r'dataset'
    csv_dir = r'csv_data'
    all_data_path = r'csv_data\all'

    train_dataset = NPC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir,
                                   set_name='train', all_data_path=all_data_path)
    val_dataset = NPC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir,
                                 set_name='validation', all_data_path=all_data_path)
    test_dataset = NPC_OS_Dataset(root=dataset_path, csv_root_path=csv_dir,
                                  set_name='test', all_data_path=all_data_path)

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)


    model = Model_TEXT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = WeightedBCELoss()

    epochs = 50
    patience = 30
    counter = 0

    best_model_path = r'best\best_model.pth'

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)

        val_loss = validate(model, val_loader, criterion)

        logger.info(
            f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val ACC: {val_auc:.4f}')

        if val_loss > best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'save best model : {val_loss:.4f}')
        else:
            counter += 1
            if counter >= patience:
                logger.info('early stop')
                break

    model.load_state_dict(torch.load(best_model_path))
    test_loss = validate(model, test_loader, criterion)
    logger.info(f'test - ACC: {test_loss:.4f}')


if __name__ == "__main__":
    main()