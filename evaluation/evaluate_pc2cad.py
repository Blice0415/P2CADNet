import h5py
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys
import torch
sys.path.append("..")
from config import ConfigAE
from trainer import TrainerAE
from cadlib.macro import *
from cadlib.extrude import CADSequence
from utils import ensure_dir
from cadlib.visualize import CADsolid2pc, create_CAD
# from pc_utils import process_pc, pc_info_save

args = {}
args["gt_z_path"] = '../proj_log/pretrained/results/all_zs_ckpt1000.h5'
args["out_z_path"] = '../proj_log/pretrained/pc2cad/results/fake_z_ckpt120_num8052.h5'
#args["out_z_path"] = '../proj_log/NewDeepCAD2/voxel2cad/results/fake_z_8052_test.h5'


with h5py.File(args["out_z_path"], "r") as fp:
    out_z = fp['zs'][:]
# print(gt_z.shape)
# print(out_z.shape)
# sum_loss = 0
# for i in range(len(out_z)):
#     loss = sum((gt_z[i] - out_z[i]) * (gt_z[i] - out_z[i]))
#     sum_loss += loss
# z_mse_info = "MSELoss: {}".format(sum_loss / len(out_z))
# print(z_mse_info)

def decode(tr_agent, zs):

    # decode
    vecs = []
    for i in range(0, len(zs)):
        with torch.no_grad():
            batch_z = torch.tensor(zs[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            batch_z = batch_z.cuda()
            outputs = tr_agent.decode(batch_z)
            batch_out_vec = tr_agent.logits2vec(outputs)
            vecs.append(batch_out_vec)
    return vecs

phas = 'test'
cfg = ConfigAE(phas)
tr_agent = TrainerAE(cfg)
tr_agent.load_ckpt(cfg.ckpt)
tr_agent.net.eval()

gt_vecs = 
out_vecs = decode(tr_agent, out_z)

TOLERANCE = 3
N_POINTS = 8096

# overall accuracy
avg_cmd_acc = [] # ACC_cmd
avg_param_acc = [] # ACC_param

# accuracy w.r.t. each command type
each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

# accuracy w.r.t each parameter
args_mask = CMD_ARGS_MASK.astype(np.float)
N_ARGS = args_mask.shape[1]
each_param_cnt = np.zeros([*args_mask.shape])
each_param_acc = np.zeros([*args_mask.shape])

gt_cad_error = 0
out_cad_error = 0
dists = []
allloss = [0] * 8
counter = 0
INVALID_INDEX = [5530]
for i in range(len(out_vecs)):
    
    if i % 100 == 0:
        print(i)

    out_vec = out_vecs[i][0]
    gt_vec = gt_vecs[i][0]

    out_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = out_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    cmd_acc = (out_cmd == gt_cmd).astype(np.int)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]
        each_cmd_cnt[cmd] += 1
        each_cmd_acc[cmd] += cmd_acc[j]
        if cmd in [SOL_IDX, EOS_IDX]:
            continue

        if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
            tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(np.int)
            # filter param that do not need tolerance (i.e. requires strictly equal)
            if cmd == EXT_IDX:
                tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(np.int)[-2:]
            elif cmd == ARC_IDX:
                tole_acc[3] = (out_param[j] == gt_param[j]).astype(np.int)[3]

            valid_param_acc = tole_acc[args_mask[cmd].astype(np.bool)].tolist()
            param_acc.extend(valid_param_acc)

            each_param_cnt[cmd, np.arange(N_ARGS)] += 1
            each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

    param_acc = np.mean(param_acc)
    avg_param_acc.append(param_acc)
    cmd_acc = np.mean(cmd_acc)
    avg_cmd_acc.append(cmd_acc)

    # pc
    # if INVALID_INDEX.count(i) == 1:
    #     continue
    # try:
    #     gt_cad_seq = CADSequence.from_vector(gt_vec, True)
    #     gt_shape = create_CAD(gt_cad_seq)
    #     gt_pc = CADsolid2pc(gt_shape, N_POINTS)
    # except Exception as e:
    #     gt_cad_error += 1
    #     continue

    # try:
    #     out_cad_seq = CADSequence.from_vector(out_vec, True)
    #     out_shape = create_CAD(out_cad_seq)
    #     out_pc = CADsolid2pc(out_shape, N_POINTS)
    # except Exception as e:
    #     out_cad_error += 1
    #     continue

    
#     loss = process_pc(gt_pc, out_pc)
#     if loss is None:
#         continue
#     res = loss[0]
#     loss = loss[1:]
#     dists.append(res)
#     for i in range(0, len(loss)):
#         allloss[i] += loss[i]
#     counter += 1

# aim = "voxel"

print("Ground Truth Vec Error: {}".format(gt_cad_error))
print("Out Vec Error: {}".format(out_cad_error))
# pc_info_save(dists, allloss, counter, "test_{}.txt".format(aim))

save_path = "{}_2cad_acc_stat.txt".format(phas)
fp = open(save_path, "w")
# overall accuracy (averaged over all data)
avg_cmd_acc = np.mean(avg_cmd_acc)
print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
avg_param_acc = np.mean(avg_param_acc)
print("avg param acc (ACC_param):", avg_param_acc, file=fp)

# acc of each command type
each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
print("each command count:", each_cmd_cnt, file=fp)
print("each command acc:", each_cmd_acc, file=fp)

# acc of each parameter type
each_param_acc = each_param_acc * args_mask
each_param_cnt = each_param_cnt * args_mask
each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
for i in range(each_param_acc.shape[0]):
    print(ALL_COMMANDS[i] + " param acc:", each_param_acc[i][args_mask[i].astype(np.bool)], file=fp)
print("z_mse: {}".format(z_mse_info), file=fp)
fp.close()

# with open(save_path, "r") as fp:
#     res = fp.readlines()
#     for l in res:
#         print(l, end='')