import numpy as np
import openpmd_api as io
from tqdm import tqdm
from pathlib import Path
from scipy.constants import m_e, c

# 获取单个文件的粒子数据
def get_particle_data(file):
    series = io.Series(file, io.Access_Type.read_only)

    x, y, z, ux, uy, uz, w, pid = [], [], [], [], [], [], [], []

    for iteration in series.iterations:
        i = series.iterations[iteration]
        electrons = i.particles["electrons"]

        # 位置
        x.append(electrons["position"]["x"][:])
        y.append(electrons["position"]["y"][:])
        z.append(electrons["position"]["z"][:])

        # 动量
        ux.append(electrons["momentum"]["x"][:])
        uy.append(electrons["momentum"]["y"][:])
        uz.append(electrons["momentum"]["z"][:])

        # 权重
        w.append(electrons["weighting"]["\x0bScalar"][:])

        # 粒子编号（id）
        pid.append(electrons["id"]["\x0bScalar"][:])

    # 触发实际数据加载
    series.flush()

    # 合并所有时间步的数据
    return np.concatenate(x), np.concatenate(y), np.concatenate(z), np.concatenate(ux), np.concatenate(uy), np.concatenate(uz), np.concatenate(w), np.concatenate(pid)

def track_particles(wkdir, selected_pid_values):
    # 用于存储符合条件的轨迹数据
    tracks = {}
    filedir = Path(wkdir)
    # 遍历目录中的所有文件，筛选出这些粒子的轨迹
    for file in tqdm(list(filedir.glob("*.h5"))):
        # 获取该文件的粒子数据
        x, y, z, ux, uy, uz, w, pid = get_particle_data(str(file))
        
        # 筛选出符合条件的粒子（在此文件中）
        selected_mask = np.isin(pid, selected_pid_values)
        
        # 提取符合条件的粒子的轨迹数据
        for i, particle_id in enumerate(pid[selected_mask]):
            # 如果这个粒子还没有被记录过，初始化它的轨迹数据并记录其起始索引
            if particle_id not in tracks:
                tracks[particle_id] = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "ux": [],
                    "uy": [],
                    "uz": [],
                    # "w": [],
                    "w": w[selected_mask][i],  # 记录权重
                    # "pid": particle_id,
                    "idx_start": i  # 记录起始索引
                }
            
            # 将粒子的轨迹数据追加到相应的轨迹中
            tracks[particle_id]["x"].append(x[selected_mask][i])
            tracks[particle_id]["y"].append(y[selected_mask][i])
            tracks[particle_id]["z"].append(z[selected_mask][i])
            tracks[particle_id]["ux"].append(ux[selected_mask][i]/m_e/c)
            tracks[particle_id]["uy"].append(uy[selected_mask][i]/m_e/c)
            tracks[particle_id]["uz"].append(uz[selected_mask][i]/m_e/c)
            # tracks[particle_id]["w"].append(w[selected_mask][i])
    return tracks