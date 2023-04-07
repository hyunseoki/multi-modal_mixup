import os, argparse, torch, timm
import pandas as pd
from metric import accuracy_function
from engine import ModelTrainer
from util import seed_everything, load_model_weights, get_sampler, str2bool
from dataset import DaconDataset, get_train_transforms, get_valid_transforms, get_test_transforms
from model import DaconLSTM, DaconModel


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='/home/hyunseoki/ssd1/01_dataset/dacon/LG_plant_disease/data/train')
    parser.add_argument('--save_folder', type=str, default='./checkpoint')
    # parser.add_argument('--kfold_idx', type=int, default=0)

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--cnn_backbone', type=str, default='')
    parser.add_argument('--rnn_backbone', type=str, default='')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--align_csv', type=str2bool, default=False)

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

    # train_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/5fold_seed42/train{args.kfold_idx}.csv')
    # valid_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/5fold_seed42/val{args.kfold_idx}.csv')
    train_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/paper_work_seed777/train.csv')
    valid_df = pd.read_csv(f'/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/paper_work_seed777/val.csv')

    train_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=train_df,
        align_csv=args.align_csv,
        transforms=get_train_transforms(),
    )

    valid_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=valid_df,
        align_csv=args.align_csv,
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
    if args.scheduler == None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, eta_min=args.lr / 1e3)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_data_loader),
            epochs=args.epochs,
        )
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

    test_df = pd.read_csv('/home/hyunseoki/ssd1/02_src/LG_plant_disease/data/paper_work_seed777/test.csv')

    test_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=test_df,
        phase='test',
        max_len=320, 
        align_csv=args.align_csv,
        transforms=get_test_transforms(),
    )

    test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

    import numpy as np
    import glob, tqdm

    results = np.zeros(shape=(len(test_df), 25))
    weight_fns = glob.glob(f'{trainer.save_path}/**/*.pth', recursive=True)
    for weight_fn in weight_fns:
        model = load_model_weights(model, weight_fn)
        print(f'{weight_fn} is loaded')
        model.to(args.device)
        model.eval()

        for idx, sample in enumerate(tqdm.tqdm(test_data_loader)):
            img, csv = sample['image'].to(args.device), sample['csv'].to(args.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(img, csv)

            batch_index = idx * args.batch_size
            results[batch_index:batch_index+args.batch_size] += output.clone().detach().cpu().numpy() ## soft-vote

    true = [test_dataset.encode(label) for label in test_df['label']]
    pred = np.array([np.argmax(result) for result in results])
    test_acc = accuracy_function(real=true, pred=pred)

    with open(os.path.join(trainer.save_dir, 'test.txt'), 'w') as f:
        f.write(f'test acc : {test_acc}') 

if __name__ == '__main__':
    main()