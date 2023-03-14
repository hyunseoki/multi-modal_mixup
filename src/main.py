import os, argparse, torch, timm
import pandas as pd
from metric import accuracy_function
from engine import ModelTrainer
from util import seed_everything, load_model_weights, get_sampler
from dataset import DaconDataset, get_train_transforms, get_valid_transforms
from model import DaconLSTM, DaconModel
from transformers.optimization import get_cosine_schedule_with_warmup


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='/home/hyunseoki/ssd1/01_dataset/dacon/LG_plant_disease/data/train')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')
    parser.add_argument('--kfold_idx', type=int, default=0)

    parser.add_argument('--model', type=str, default='tf_efficientnetv2_s')
    parser.add_argument('--cnn_backbone', type=str, default='')
    parser.add_argument('--rnn_backbone', type=str, default='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    assert os.path.isdir(args.base_folder), f'wrong path {args.base_folder}'
    if args.comments is not None:
        args.save_folder = os.path.join(args.save_folder, args.comments)

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)  

    train_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/5fold_seed42/train{args.kfold_idx}.csv')
    valid_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/5fold_seed42/val{args.kfold_idx}.csv')

    train_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    valid_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=valid_df,
        transforms=get_valid_transforms(),
    )

    train_sampler = get_sampler(
        df=train_df,
        dataset=train_dataset
    )

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            # shuffle=True,
            num_workers=8,
        )

    valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
        )

    model_cnn = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=25
    )
    if args.cnn_backbone != '':
        model_cnn = load_model_weights(model=model_cnn, weight_fn=args.cnn_backbone)
        cnn_fn = args.cnn_backbone
        print(f'[info msg] pre-trained cnn weight {cnn_fn} is loaded')
        print('=' * 50) 

    rnn_model = DaconLSTM()
    if args.rnn_backbone != '':
        rnn_fn = args.rnn_backbone
        print(f'[info msg] pre-trained cnn weight {rnn_fn} is loaded')
        print('=' * 50) 
        rnn_model = load_model_weights(model=rnn_model, weight_fn=rnn_fn)

    model = DaconModel(
        model_cnn=model_cnn,
        model_rnn=rnn_model
    )
    
    loss = torch.nn.CrossEntropyLoss()
    metric = accuracy_function
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=args.lr / 1e3)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=1149,
    #     num_training_steps=int(len(train_dataset) * args.epochs/args.batch_size),
    # )
    trainer = ModelTrainer(
            model=model,
            train_loader=train_data_loader,
            valid_loader=valid_data_loader,
            loss_func=loss,
            metric_func=metric,
            optimizer=optimizer,
            device=args.device,
            save_dir=args.save_folder,
            mode='max', 
            scheduler=scheduler, 
            num_epochs=args.epochs,
            num_snapshops=None,
            parallel=False,
            use_csv=True,
            use_wandb=False,
        )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()