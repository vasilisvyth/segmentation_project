from pathlib import Path
from loss import DiceLoss
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from scaling import standardize, normalize
from lighting_model import AtriumSegmentation
from dataset import CardiacDataset
from model import UNet, AttentionUNet
import torch.utils.data as data

def main(args):
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.85, 1.15),
                rotate=(-45, 45)),
        iaa.ElasticTransformation()
    ])

    # Create the dataset objects
    train_path = Path("Preprocessed/train/")
    val_path = Path("Preprocessed/val")

    train_dataset = CardiacDataset(train_path, seq)
    val_dataset = CardiacDataset(val_path, None)

    print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")

    batch_size = args.batch_size
    num_workers = 1

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    torch.manual_seed(0)
    if args.model.lower()=='unet':
        segmentation_model = UNet()
    elif args.model.lower()=='attention_unet':
        segmentation_model = AttentionUNet()
    else:
        raise ValueError(f'We do not support {args.model}')
    model = AtriumSegmentation(segmentation_model)


    checkpoint_callback = ModelCheckpoint(
        monitor='Val Dice',
        save_top_k=10,
        mode='min')


    gpus = 0#1 #TODO
    trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
                        callbacks=checkpoint_callback,max_epochs=2,#75
                        limit_train_batches=1,  # Limit to a single training batch per epoch
                        limit_val_batches=1)


    trainer.fit(model, train_loader, val_loader)


    import nibabel as nib
    from tqdm.notebook import tqdm
    from celluloid import Camera


    model = AtriumSegmentation.load_from_checkpoint("weights/70.ckpt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    preds = []
    labels = []

    for slice, label in tqdm(val_dataset):
        slice = torch.tensor(slice).to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(slice)
        preds.append(pred.cpu().numpy())
        labels.append(label)
        
    preds = np.array(preds)
    labels = np.array(labels)


    loss = 1-model.loss_fn(torch.from_numpy(preds), torch.from_numpy(labels))  
    print(f'loss {loss}')

    dice_score = 1-DiceLoss()(torch.from_numpy(preds), torch.from_numpy(labels).unsqueeze(0).float())
    print(f"The Val Dice Score is: {dice_score}")


    # ## Visualization

    # We can now load a test subject from the dataset and estimate the position of the left atrium

    subject = Path("Task02_Heart/imagesTs/la_002.nii.gz")
    subject_mri = nib.load(subject).get_fdata()

    # As this scan is neither normalized nor standardized we need to perform those tasks!<br />
    # Let us copy the normalization and standardization functions from our preprocessing notebook:

    subject_mri = subject_mri[32:-32, 32:-32]
    standardized_scan = standardize(normalize(subject_mri))

    preds = []
    for i in range(standardized_scan.shape[-1]):
        slice = standardized_scan[:,:,i]
        with torch.no_grad():
            pred = model(torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device))[0][0]
            pred = pred > 0.5
        preds.append(pred.cpu())

    from animation import animate
    animate(standardized_scan, preds, 'mri_animation.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # Add arguments
    # parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels (e.g., 1 for grayscale, 3 for RGB)')
    # parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels (e.g., 1 for binary segmentation)')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--model',type=str, default='attention_unet')
    # parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    # parser.add_argument('--data_path', type=str, default='./data', help='Path to the training data')
    # parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)