from tqdm import tqdm
import argparse
from pc2cad2 import Config,TrainAgent,get_dataloader
import torch
import numpy as np
import os
import h5py
from cadlib.macro import EOS_IDX
from utils import cycle, ensure_dirs, ensure_dir, read_ply, write_ply


def main():
    # create experiment cfg containing all hyperparameters
    # print("1111111")
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--proj_dir', type=str, default="proj_log",
                    help="path to project folder where models and logs will be saved")
    parser1.add_argument('--raw_root', type=str, default="path_to_data", help="path to data folder1")
    parser1.add_argument('--pc_root', type=str, default="path_to_pc_data", help="path to point clouds data folder")
    parser1.add_argument('--split_path', type=str, default="data/train_val_test_split.json", help="path to train-val-test split")
    parser1.add_argument('--exp_name', type=str, required=True, help="name of this experiment")
    parser1.add_argument('--ae_ckpt', type=str, required=True, help="desired checkpoint to restore")
    parser1.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
    parser1.add_argument('--ckptpc', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser1.add_argument('--test',action='store_true', help="test mode")
    parser1.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
    parser1.add_argument('-g', '--gpu_ids', type=str, default="0",
                    help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
    args = parser1.parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    cfgpc = Config(args)
    print("data path:", cfgpc.data_root)
    agent = TrainAgent(cfgpc)
    # create network and training agent

    # load from checkpoint if provided
    # load trained weights
    agent.load_ckpt(args.ckptpc)

    agent.net.eval()

    test_loader = get_dataloader('test', cfgpc)

    save_dir = os.path.join(cfgpc.exp_dir, "results/test_{}".format(args.ckptpc))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        batch_size = data['command'].shape[0]
        commands = data['command']
        arg = data['args']
        gt_vec = torch.cat([commands.unsqueeze(-1), arg], dim=-1).squeeze(1).detach().cpu().numpy()
        gt_vec = gt_vec[:,:60,:]
        commands_ = gt_vec[:, :, 0]
        with torch.no_grad():
            outputs, _ = agent.forward(data)
            batch_out_vec = agent.logits2vec(outputs)
            
        pts = data['points'].detach().cpu().numpy()
        for j in range(batch_size):
            save_path = os.path.join(save_dir, "{}.ply".format(data['id'][j]))
            truck_dir = os.path.dirname(save_path)
            if not os.path.exists(truck_dir):
                os.makedirs(truck_dir)
            write_ply(pts[j], save_path)
            out_vec = batch_out_vec[j][:60]
            # print(gt_vec.shape)
            seq_len = commands_[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split('/')[-1]

            save_path = os.path.join(cfgpc.exp_dir, 'results/vec/{}_vec.h5'.format(data_id))
            truck_dir = os.path.dirname(save_path)
            if not os.path.exists(truck_dir):
                os.makedirs(truck_dir)
            with h5py.File(save_path, 'w') as fp:
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
                fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)


if __name__ == '__main__':
    main()
