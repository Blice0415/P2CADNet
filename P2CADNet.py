import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import h5py
import shutil
import json
import random
import sys


sys.path.append("..")
from trainer.base import BaseTrainer
from model.autoencoder import Decoder, Bottleneck
from utils import cycle, ensure_dirs, ensure_dir, read_ply, write_ply
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from model.model_utils import _make_batch_first
from cadlib.macro import *
from trainer.loss import CADLoss


class Config(object):
    n_points = 2048
    batch_size = 16
    num_workers = 0
    nr_epochs = 200
    lr = 0.0002
    lr_step_size = 200
    # beta1 = 0.5
    grad_clip = None

    save_frequency = 20
    val_frequency = 10

    def __init__(self, args):
        self.data_root = os.path.join(
            args.proj_dir, args.exp_name,
            "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        self.pc_root = args.pc_root
        self.raw_root = args.raw_root
        self.split_path = args.split_path
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "pc2cad")
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.gpu_ids = args.gpu_ids
        self.args_dim = ARGS_DIM  # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)
        self.max_total_len = MAX_TOTAL_LEN
        self.loss_weights = {"loss_cmd_weight": 1.0, "loss_args_weight": 2.0}
        self.n_heads = 8  # Transformer config: number of heads
        self.dim_feedforward = 512  # Transformer config: FF dimensionality
        self.d_model = 256  # Transformer config: model dimensionality
        self.dropout = 0.1  # Dropout rate used in basic layers and Transformers
        self.dim_z = 256
        self.n_layers_decode = 4

        if (not args.test) and args.cont is not True and os.path.exists(
                self.exp_dir):
            response = input(
                'Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            # shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if not args.test:
            os.system("cp pc2cad.py {}".format(self.exp_dir))
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)


class PointNet2Vec(nn.Module):

    def __init__(self, config):
        super(PointNet2Vec, self).__init__()

        self.use_xyz = True

        self.config = config
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[0, 32, 32, 64],
                # bn=False,
                use_xyz=self.use_xyz,
            ))

        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                # bn=False,
                use_xyz=self.use_xyz,
            ))

        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                # bn=False,
                use_xyz=self.use_xyz,
            ))

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                # bn=False,
                use_xyz=self.use_xyz))

        self.fc_layer = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(True),
                                      nn.Linear(512, 256), nn.LeakyReLU(True),
                                      nn.Linear(256, 256), nn.Tanh())
        
        self.bottleneck = Bottleneck(self.config)

        self.decoder = Decoder(self.config)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, commands_enc, args_enc):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        z = self.fc_layer(features.squeeze(-1)).unsqueeze(0)
        z = self.bottleneck(z)
        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)
        res = {"command_logits": out_logits[0], "args_logits": out_logits[1]}
        res["tgt_commands"] = commands_enc
        res["tgt_args"] = args_enc
        return res


class TrainAgent(BaseTrainer):

    def build_net(self, config):
        self.net = PointNet2Vec(config).cuda()

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), config.lr)  # , betas=(config.beta1, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate',
                                 self.optimizer.param_groups[-1]['lr'],
                                 self.clock.epoch)
        self.scheduler.step()

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'],
                                                 dim=-1),
                                   dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1),
                                dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad:  # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[
                out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return loss
    
    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        loss = self.update_network(losses)
        if self.clock.step % 10 == 0:
            self.record_losses(losses, 'train')

        return outputs, losses, loss

    def forward(self, data):
        points = data["points"].cuda()
        commands = data["command"].cuda()
        args = data["args"].cuda()

        outputs = self.net(points, commands, args)
        # for k,v in outputs.items():
        #     print(k,":",v.size())

        loss_dict = self.loss_func(outputs)
        return outputs, loss_dict


class ShapeCodesDataset(Dataset):

    def __init__(self, phase, config):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.data_root = config.data_root
        self.raw_data = os.path.join(config.raw_root,
                                     "cad_vec")  # h5 data root
        self.pc_root = config.pc_root
        self.path = config.split_path
        self.max_total_len = config.max_total_len
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + '.ply')
        # print(pc_path)
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc = read_ply(pc_path)
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        pc = torch.tensor(pc, dtype=torch.float32)

        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate(
            [cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"points": pc, "command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = ShapeCodesDataset(phase, config)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=is_shuffle,
                            num_workers=config.num_workers)
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proj_dir',
        type=str,
        default="proj_log",
        help="path to project folder where models and logs will be saved")
    parser.add_argument('--pc_root',
                        type=str,
                        default="path_to_pc_data",
                        help="path to point clouds data folder")
    parser.add_argument('--raw_root',
                        type=str,
                        default="data",
                        help="path to data folder")
    parser.add_argument('--split_path',
                        type=str,
                        default="data/train_val_test_split.json",
                        help="path to train-val-test split")
    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help="name of this experiment")
    parser.add_argument('--ae_ckpt',
                        type=str,
                        required=True,
                        help="desired checkpoint to restore")
    parser.add_argument('--continue',
                        dest='cont',
                        action='store_true',
                        help="continue training from checkpoint")
    parser.add_argument('--ckpt',
                        type=str,
                        default='latest',
                        required=False,
                        help="desired checkpoint to restore")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--n_samples',
                        type=int,
                        default=100,
                        help="number of samples to generate when testing")
    parser.add_argument('-g',
                        '--gpu_ids',
                        type=str,
                        default="0",
                        help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
    args = parser.parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    cfg = Config(args)
    print("data path:", cfg.data_root)
    agent = TrainAgent(cfg)

    if not args.test:
        # load from checkpoint if provided
        if args.cont:
            agent.load_ckpt(args.ckpt)

        # create dataloader
        train_loader = get_dataloader('train', cfg)
        val_loader = get_dataloader('validation', cfg)
        val_loader = cycle(val_loader)

        # start training
        clock = agent.clock

        for e in range(clock.epoch, cfg.nr_epochs):
            # begin iteration
            pbar = tqdm(train_loader)
            loss_sum = 0
            for b, data in enumerate(pbar):
                # train step
                outputs, losses, loss = agent.train_func(data)
                loss_sum += loss
                loss_mean = loss_sum/b
                losses["loss_mean"] = loss_mean
                pbar.set_description("EPOCH[{}][{}]".format(e, b))
                pbar.set_postfix({k: v.item() for k, v in losses.items()})

                # validation step
                # if clock.step % cfg.val_frequency == 0:
                #     data = next(val_loader)
                #     outputs, losses = agent.val_func(data)

                clock.tick()

            clock.tock()

            if clock.epoch % cfg.save_frequency == 0:
                agent.save_ckpt()

            # if clock.epoch % 10 == 0:
            agent.save_ckpt('latest')
    else:
        # load trained weights
        agent.load_ckpt(args.ckpt)

        agent.net.eval()

        test_loader = get_dataloader('test', cfg)

        save_dir = os.path.join(
            cfg.exp_dir,
            "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_zs = []
        pbar = tqdm(test_loader)
        cnt = 0
        for i, data in enumerate(pbar):
            batch_size = data['command'].shape[0]
            commands = data['command']
            args = data['args']
            gt_vec = torch.cat([commands.unsqueeze(-1), args],
                            dim=-1).squeeze(1).detach().cpu().numpy()
            commands_ = gt_vec[:, :, 0]
            with torch.no_grad():
                outputs, _ = agent.forward(data)
                batch_out_vec = agent.logits2vec(outputs)
            for j in range(batch_size):
                out_vec = batch_out_vec[j]
                seq_len = commands_[j].tolist().index(EOS_IDX)

                data_id = data["id"][j].split('/')[-1]

                save_path = os.path.join(cfg.outputs, '{}_vec.h5'.format(data_id))
                with h5py.File(save_path, 'w') as fp:
                    fp.create_dataset('out_vec',
                                    data=out_vec[:seq_len],
                                    dtype=np.int)
                    fp.create_dataset('gt_vec',
                                    data=gt_vec[j][:seq_len],
                                    dtype=np.int)
            pts = data['points'].detach().cpu().numpy()
            for j in range(batch_size):
                save_path = os.path.join(save_dir, "{}.ply".format(data['id'][j]))
                truck_dir = os.path.dirname(save_path)
                if not os.path.exists(truck_dir):
                    os.makedirs(truck_dir)
                write_ply(pts[j], save_path)
            cnt += batch_size
            if cnt > args.n_samples:
                break


if __name__ == '__main__':
    main()
