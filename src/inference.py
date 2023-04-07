import torch
import os
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import timm
from util import seed_everything, load_model_weights, str2bool
from dataset import DaconDataset, get_test_transforms
from model import DaconLSTM, DaconModel


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='/home/hyunseoki/ssd1/01_dataset/dacon/LG_plant_disease/data/test')
    parser.add_argument('--save_folder', type=str, default='./submission')
    parser.add_argument('--weight_folder', type=str, default='/home/hyunseoki/ssd1/02_src/LG_plant_disease/checkpoint/baseline_scratch')
    parser.add_argument('--label_fn', type=str, default='./data/sample_submission.csv')
    parser.add_argument('--model', type=str, default='tf_efficientnetv2_s')
    parser.add_argument('--csv_align', type=str2bool, default=False)

    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--comments', type=str, default='')

    args = parser.parse_args()

    assert os.path.isdir(args.base_folder), 'wrong path'
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    assert os.path.isdir(args.weight_folder), 'wrong path'
    assert os.path.isfile(args.label_fn), 'wrong path'

    test_df = pd.read_csv(args.label_fn)

    test_dataset = DaconDataset(
        base_folder=args.base_folder,
        label_df=test_df,
        phase='test',
        max_len=320, 
        align_csv=args.csv_align,
        transforms=get_test_transforms(),
    )

    test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

    model = DaconModel(
        model_cnn=timm.create_model(args.model, pretrained=False, num_classes=25),
        model_rnn=DaconLSTM()
    )

    # weight_fns = glob.glob(args.weight_folder + '/*.pth')
    weight_fns = glob.glob(f'{args.weight_folder}/**/*.pth', recursive=True)
    assert len(weight_fns) == 5, 'weight가 5개 미만임'

    results = np.zeros(shape=(len(test_df), 25))

    for weight_fn in weight_fns:
        model = load_model_weights(model, weight_fn)
        print(f'{weight_fn} is loaded')
        model.to(device)
        model.eval()

        for idx, sample in enumerate(tqdm(test_data_loader)):
            img, csv = sample['image'].to(device), sample['csv'].to(device)
            with torch.no_grad():
                output = model(img, csv)

            batch_index = idx * args.batch_size
            results[batch_index:batch_index+args.batch_size] += output.clone().detach().cpu().numpy() ## soft-vote

    voting_results = np.array([test_dataset.decode(np.argmax(result)) for result in results])
    test_df['label'] = voting_results

    if args.comments == '':
        from datetime import datetime
        save_fn = str(os.path.join(args.save_folder, datetime.now().strftime("%m%d%H%M%S"))) + '.csv'
    else:
        save_fn = str(os.path.join(args.save_folder, args.comments)) + '.csv'
    test_df.to_csv(save_fn, index=False)
    print(f'{save_fn} is saved')


if __name__ == '__main__':
    main()