import os
import glob
import json
import numpy as np
import random
import h5py
from joblib import Parallel, delayed
from trimesh.sample import sample_surface
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2mesh, create_CAD
from utils.pc_utils import write_ply, read_ply
import trimesh

DATA_ROOT = "../data"
RAW_DATA = os.path.join(DATA_ROOT, "cad_json")
RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")
# count = 0
N_POINTS = 8192 # 4096
WRITE_NORMAL = False
SAVE_DIR = os.path.join(DATA_ROOT, "mesh_cad123123")
SAVE_DIR2 = os.path.join(DATA_ROOT, "mesh_cad_500123213")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

INVALID_IDS = ["0011/00116212","0042/00426282"]

max_edge = 0

def process_one(data_id):
    # global all_data
    # path = os.path.join(SAVE_DIR, data_id + ".npz")
    # if(not os.path.exists(path)):
        # all_data[phase].remove(data_id)
    if data_id in INVALID_IDS:
        print("skip {}: in invalid id list".format(data_id))
        return
    print("processing:",data_id)
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        print("skip {}: file already exists".format(data_id))
        return

    print("[processing] {}".format(data_id))
    json_path = os.path.join(RAW_DATA, data_id + ".json")
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        print("create_CAD failed:", data_id)
        return None

    try:
        out_mesh = CADsolid2mesh(shape, N_POINTS, data_id.split("/")[-1])
    except Exception as e:
        print("convert mesh failed:", data_id)
        return None

    save_path = os.path.join(SAVE_DIR, data_id + ".obj")
    save_path2 = os.path.join(SAVE_DIR2, data_id + ".obj")
    truck_dir = os.path.dirname(save_path)
    truck_dir2 = os.path.dirname(save_path2)
    # if(os.path.exists(save_path) and not os.path.exists(save_path2)):
    #     print(data_id,count)
    # count += 1
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)
    if not os.path.exists(truck_dir2):
        os.makedirs(truck_dir2)
    # print(out_mesh.edges)
    out_mesh.export(file_obj=save_path,file_type='obj')
    command = "\"C:\\Program Files\\Blender Foundation\\Blender 3.1\\blender.exe\" -b -P E:\\MeshCNN-master\\scripts\\dataprep\\blender_process.py " + save_path + " 500 " + save_path2
    os.system(command)
    # # mesh = trimesh.load(save_path)
    # if(os.path.exists(save_path2)):
    #     mesh = trimesh.load(save_path2)
    #     print(mesh.is_watertight)
    #     if(not mesh.is_watertight):
    #         # os.remove(save_path2)
    #         print(data_id)
        # command = "\"E:\\manifold-master\\build\\Release\\manifold.exe\" " + save_path + " " + save_path2 + " 10000"
        # print(len(mesh.edges))
        # os.system(command)
        # print(len(mesh.edges))
        # if(len(mesh.edges)>max_edge):
            # max_edge = len(mesh.edges)
            # os.remove(save_path2)
            # print(data_id)
            # command = "\"E:\\manifold-master\\build\\Release\\simplify.exe\" -i " + save_path2 + " -o " + save_path2 + " -m -c 1e-2 -f 500"
            # print(command)
            # os.system(command)
    # print(data_id, mesh.is_watertight)



with open(RECORD_FILE, "r") as fp:
    all_data = json.load(fp)

# process_one(all_data["train"][3])
# exit()

parser = argparse.ArgumentParser()
parser.add_argument('--only_test', action="store_true", help="only convert test data")
args = parser.parse_args()
if not args.only_test:
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
# print(len(all_data["train"]))
# for x in all_data["train"]:
#     process_one(x, "train")
# for x in all_data["test"]:
#     process_one(x, "test")
# for x in all_data["validation"]:
#     process_one(x, "validation")
# with open(r'E:\MeshNet-master\dataset\train_val_test_split2.json',"w") as js:
#     data = json.dumps(all_data)
#     js.write(data)

# print(max_edge)
# process_one("0018/00180328")
# process_one("0000/00000070")
# process_one("0000/00000093")
# process_one("0042/00426282")
