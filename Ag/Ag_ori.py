import os
import sys,re
import tempfile
import numpy as np
import subprocess
import multiprocessing
import glob
import shutil
import math
import time
import warnings
from dataclasses import dataclass
from botorch.test_functions.synthetic import SyntheticTestFunction
import torch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from abc import ABC, abstractmethod
import gpytorch
from botorch.acquisition import qUpperConfidenceBound
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import List, Optional, Tuple
import IPython.display as display
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from botorch.utils.transforms import normalize 
from botorch.utils.transforms import unnormalize

from iocode.tensor import read_and_process_data, write_tensor_to_file, load_tensor_from_file
from iocode.iofile import read_POSCAR, read_vibrocc, read_poscar, write_poscar, batch_modify_poscar_data, remove_tensors_folder,update_theta, read_and_update_config
from iocode.iolog import get_newest_log_file, extract_r_value_from_log, write_parameters_file

# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float
dtype = torch.double
print(device)
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# os.system('export OMP_NUM_THREADS=48')
# os.system('ulimit -v 48000000')
root_path = os.getcwd()+'/'

def execute_in_folder(folder_path, shell_script_name):

    try:
       
        original_cwd = os.getcwd()
        os.chdir(folder_path)
       
        shell_script_path = os.path.join(folder_path, shell_script_name)
        os.chmod(shell_script_path, 0o755)
       
        print(f"进程 {os.getpid()}: 在文件夹 {folder_path} 中执行 {shell_script_name}")
        
        result = subprocess.run(
            [f'./{shell_script_name}'],
            capture_output=True,
            text=True,
            timeout=300  
        )
        read_and_update_config('PARAMETERS',updates ={"RUN":1})
        
        result = subprocess.run(
            [f'./{shell_script_name}'],
            capture_output=True,
            text=True,
            timeout=600  
        )
  
        os.chdir(original_cwd)
        
        return {
            'folder': folder_path,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        return {
            'folder': folder_path,
            'returncode': -1,
            'stdout': '',
            'stderr': '执行超时',
            'success': False
        }
    except Exception as e:
        os.chdir(original_cwd)
        return {
            'folder': folder_path,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }
    
def find_ordered_y_pairs(y_coords, tolerance=1e-6):
    """
    Check consecutive pairs: (0,1), (2,3), (4,5), etc.
    Identify which pairs have y-coordinates differing by 0.5
    """
    n_atoms = len(y_coords)
    y_rounded = np.round(y_coords, decimals=6)
    
    paired_indices = []  # List of (smaller_idx, larger_idx) tuples
    unpaired_indices = []  # List of atom indices that don't have the 0.5 relationship
    
    # Check pairs: (0,1), (2,3), (4,5), ...
    for i in range(0, n_atoms-1, 2):
        y1 = y_rounded[i]
        y2 = y_rounded[i+1]
        
        # Check if they differ by 0.5 (either direction)
        diff = abs(y1 - y2)
        
        # Also check wrap-around case: |y1 - y2| ≈ 0.5
        diff_wrapped = min(diff, 1.0 - diff)  # Consider periodic boundary
        
        if abs(diff_wrapped - 0.5) < tolerance:
            # They form a pair - determine which is smaller
            if y1 < y2:
                smaller_idx, larger_idx = i, i+1
            else:
                smaller_idx, larger_idx = i+1, i
            paired_indices.append((smaller_idx, larger_idx))
            print(f"Pair found: atoms {smaller_idx},{larger_idx} with y={y_rounded[smaller_idx]:.6f}, {y_rounded[larger_idx]:.6f}")
        else:
            # They don't form a 0.5-relationship pair
            unpaired_indices.extend([i, i+1])
            print(f"No pair: atoms {i},{i+1} with y={y1:.6f}, {y2:.6f} (diff={diff:.6f})")
    
    # Handle odd number of atoms
    if n_atoms % 2 == 1:
        unpaired_indices.append(n_atoms - 1)
        print(f"Unpaired (odd): atom {n_atoms-1} with y={y_rounded[n_atoms-1]:.6f}")
    
    return paired_indices, unpaired_indices

def create_y_optimization_mapping(n_atoms, paired_indices, unpaired_indices):
    """
    Create mapping from atom index to optimization variable index
    - For paired atoms: only the smaller y-coordinate is optimized
    - For unpaired atoms: each gets its own optimization variable
    
    Returns:
    - y_opt_mapping: array where y_opt_mapping[atom_idx] = optimization_variable_index
    - is_larger_in_pair: boolean array, True if atom uses smaller_y + 0.5
    """
    y_opt_mapping = np.zeros(n_atoms, dtype=int)
    is_larger_in_pair = np.zeros(n_atoms, dtype=bool)
    
    opt_idx = 0
    
    # Handle paired atoms
    for smaller_idx, larger_idx in paired_indices:
        # Both atoms in the pair use the same optimization variable (the smaller y)
        y_opt_mapping[smaller_idx] = opt_idx
        y_opt_mapping[larger_idx] = opt_idx
        
        # Mark which one is the larger (gets +0.5)
        is_larger_in_pair[larger_idx] = True
        
        opt_idx += 1
    
    # Handle unpaired atoms
    for atom_idx in unpaired_indices:
        y_opt_mapping[atom_idx] = opt_idx
        opt_idx += 1
    
    return y_opt_mapping, is_larger_in_pair, opt_idx

def read_POSCAR(filename='POSCAR'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    scale = float(lines[1])
    lattice = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale
    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    total_atoms = sum(atom_counts)

    coord_start = 8 if lines[7][0].lower() in ['d', 'c'] else 7
    coords = np.array([list(map(float, lines[i].split()[:3])) for i in range(coord_start, coord_start + total_atoms)])
    coord_type = lines[7][0].lower() if lines[7][0].isalpha() else 'd'

    return lattice, atom_types, atom_counts, coords, coord_type

def frac_to_cart(frac_coords, lattice):
    return np.dot(frac_coords, lattice)

def cart_to_frac(cart_coords, lattice):
    return np.dot(cart_coords, np.linalg.inv(lattice))

def read_vibrocc(filename='VIBROCC'):
    amplitudes = {}
    with open(filename, 'r') as f:
        for line in f:
       
            if '=' in line and not line.strip().startswith('='):
                key, value = line.split('=')
                key = key.strip()
   
                value = float(value.strip())
                amplitudes[key] = value
            # elif line.strip().startswith('='):
            #     break  # Stop at first non-amplitude section
    return amplitudes

lattice, atom_types, atom_counts, coords, coord_type = read_POSCAR("/work/scratch/xt17pyku/LEEDtest/Ag/pos/POSCAR")
if coord_type == 'c':
    cart_coords = coords
    frac_coords = cart_to_frac(cart_coords, lattice)
else:
    frac_coords = coords
    cart_coords = frac_to_cart(frac_coords, lattice)

# Select atoms with fractional z > 0.25
mask = frac_coords[:, 2] > 0.45
selected_frac = frac_coords[mask]
selected_cart = cart_coords[mask]

print(f"Number of selected atoms: {len(selected_cart)}")

# Apply ±0.5 Å Cartesian perturbation
delta = 0.3
lower_cart = selected_cart - delta
upper_cart = selected_cart + delta

# Convert bounds to fractional
lower_frac = cart_to_frac(lower_cart, lattice)
upper_frac = cart_to_frac(upper_cart, lattice)

# Build x,y arrays interleaved
xy_low = lower_frac[:, :2].flatten()
xy_up = upper_frac[:, :2].flatten()

# Get z arrays and remove duplicates (round to avoid floating error)
z_low, idx = np.unique(np.round(lower_frac[:, 2], 6), return_index=True)
z_low = z_low[np.argsort(idx)]  #
z_up, ix = np.unique(np.round(upper_frac[:, 2], 6), return_index=True)
z_up = z_up[np.argsort(ix)]  #

# Combine
low_vector = np.concatenate((xy_low, z_low))
up_vector = np.concatenate((xy_up, z_up))

vib_data = read_vibrocc('VIBROCC')
target_keys = ['Ag_def']
delta = 0.05

low = []
up = []

for key in target_keys:
    if key in vib_data:
        amp = vib_data[key]
        low.append(amp - delta)
        up.append(amp + delta)
    else:
        raise KeyError(f"{key} not found in VIBROCC file.")
final_low_vector = np.concatenate((low_vector, low))
final_up_vector = np.concatenate((up_vector, up))

def fileM(X, filepath,poscar,vibrocc, paras):    
    X = X.squeeze(0)
    Num_x = X
    x1 = Num_x[0]
    y1 = Num_x[1]
    x2 = Num_x[2]
    y2 = Num_x[3]
    x3 = Num_x[4]
    y3 = Num_x[5]
    x4 = Num_x[6]
    y4 = Num_x[7]
    
    z1 = Num_x[8]
    z2 = Num_x[9]
    z3 = Num_x[10]
    z4 = Num_x[11]

    Ag = Num_x[12]
    # Fe = Num_x[51]
    # theta = Num_x[52]
    data = read_poscar(filepath)
    modifications = {
        0: [x1, y1, z1],  
        1: [x2, y2, z2],   
        2: [x3, y3, z3],  
        3: [x4, y4, z4],  
    }
    modified_data = batch_modify_poscar_data(data, modifications)
    write_poscar(modified_data, poscar)
    read_and_update_config(vibrocc,updates = {"Ag_surf":Ag})

def parallel_shell_execution(X, parameters_file, shell_script_file, VIBROCC, POSCAR, EXPBEAMS,  num_processes=4):
    
    param_subsets = torch.split(X, 1, dim=0)

    filepath = "/work/scratch/xt17pyku/LEEDtest/Ag/pos/POSCAR"
    current_dir = os.getcwd()
    base_temp_dir = tempfile.mkdtemp(prefix="parallel_exec_", dir=current_dir)
    folder_paths = []

    all_successful = True
    
    try:
       
        for i in range(num_processes):
            folder_name = f"process_{i+1}"
            folder_path = os.path.join(base_temp_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            folder_paths.append(folder_path)

        shell_script_name = os.path.basename(shell_script_file)
        parameters_name = os.path.basename(parameters_file)
        VIBROCC_name = os.path.basename(VIBROCC)
        POSCAR_name = os.path.basename(POSCAR)
        EXPBEAMS_name = os.path.basename(EXPBEAMS)
        # IVBEAMS_name = os.path.basename(IVBEAMS)

        for i, folder_path in enumerate(folder_paths):
            shutil.copy2(parameters_file, os.path.join(folder_path, parameters_name))
            shutil.copy2(shell_script_file, os.path.join(folder_path, shell_script_name))
            shutil.copy2(POSCAR, os.path.join(folder_path, POSCAR_name))
            shutil.copy2(VIBROCC, os.path.join(folder_path, VIBROCC_name))
            shutil.copy2(EXPBEAMS, os.path.join(folder_path, EXPBEAMS_name))
            # shutil.copy2(IVBEAMS, os.path.join(folder_path, IVBEAMS_name))
            current_params = param_subsets[i] 
           
            fileM(current_params, filepath, os.path.join(folder_path, POSCAR_name), 
                 os.path.join(folder_path, VIBROCC_name), os.path.join(folder_path, parameters_name))
        
        start_time = time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = [(folder_path, shell_script_name) for folder_path in folder_paths]
            results = pool.starmap(execute_in_folder, tasks)
        end_time = time.time()
        
        r_values = []
        final_results = []
        any_error = False  # 记录是否有任何错误发生
        
        # 处理每个进程的结果
        for i, result in enumerate(results, 1):
            print(f"\n--- 进程 {i} 结果 ---")
            print(f"文件夹: {result['folder']}")
            print(f"返回码: {result['returncode']}")
            print(f"执行成功: {result['success']}")
                        
            # 打印输出和错误（只显示前500字符）
            if result['stdout']:
                print(f"OUTPUT:\n{result['stdout'][:500]}...")
            if result['stderr']:
                print(f"ERROR:\n{result['stderr'][:500]}...")
            
            log_result = None
            process_has_error = False  # 当前进程是否有错误
            
            if result['success']:
                newest_log = get_newest_log_file(result['folder'])
                if newest_log:
                    log_result = extract_r_value_from_log(newest_log)
                    if log_result['success']:
                        # 检查 R 值是否为 0
                        if log_result['r_float'] == 0:
                            print("警告: R 值为 0，视为失败情况")
                            log_result = {
                                'log_file': newest_log,
                                'r_value': "1000 (R=0)",
                                'r_float': 1000.0,
                                'success': False,
                                'error': 'R_VALUE_ZERO'
                            }
                            process_has_error = True
                        else:
                            print(f"R: {log_result['r_value']}")
                            print(f"FLOAT R: {log_result['r_float']}")
                            r_values.append(log_result['r_float'])
                    else:
                        # 提取日志值失败
                        print(f"警告: 日志解析失败 - {log_result.get('error', '未知错误')}")
                        log_result = {
                            'log_file': newest_log,
                            'r_value': "1000 (解析失败)",
                            'r_float': 1000.0,
                            'success': False,
                            'error': 'EXTRACTION_FAILED'
                        }
                        process_has_error = True
                else:
                    # 找不到日志文件
                    print(f"警告: 未找到日志文件")
                    log_result = {
                        'log_file': None,
                        'r_value': "1000 (无日志)",
                        'r_float': 1000.0,
                        'success': False,
                        'error': 'NO_LOG'
                    }
                    process_has_error = True
            else:
                # 进程执行失败
                print(f"警告: 进程执行失败 - 返回码 {result['returncode']}")
                log_result = {
                    'log_file': None,
                    'r_value': "1000 (执行失败)",
                    'r_float': 1000.0,
                    'success': False,
                    'error': 'EXECUTION_FAILED'
                }
                process_has_error = True
            
            # 如果当前进程有错误，标记整体错误状态
            if process_has_error:
                any_error = True
                all_successful = False
            
            combined_result = {
                'process_id': i,
                'folder': result['folder'],
                'execution': result,
                'log_analysis': log_result,
                'has_error': process_has_error
            }
            final_results.append(combined_result)
            
            # 清除显示输出（如果有）
            if not process_has_error:
                display.clear_output(wait=True)
        
        # 收集所有成功的R值（成功且R值不为0）
        successful_r_values = []
        for result in final_results:
            if result['log_analysis']['success'] and result['log_analysis']['r_float'] != 0:
                successful_r_values.append({
                    'process_id': result['process_id'],
                    'r_value': result['log_analysis']['r_float'],
                    'log_file': result['log_analysis']['log_file']
                })
            else:
                # 对于失败进程，添加R=1000的结果
                successful_r_values.append({
                    'process_id': result['process_id'],
                    'r_value': 1000.0,
                    'log_file': None
                })
        
        # 显示结果汇总
        print("\n=== 所有进程结果汇总 ===")
        if successful_r_values:
            print(f"成功提取到 {len([rv for rv in successful_r_values if rv['r_value'] < 1000])} 个有效R值:")
            for rv in successful_r_values:
                if rv['r_value'] < 1000:
                    print(f"  进程 {rv['process_id']}: R = {rv['r_value']}")
                else:
                    print(f"  进程 {rv['process_id']}: R = 1000 (失败)")
            
            # 找出最佳R值（排除1000的值）
            valid_r_values = [rv for rv in successful_r_values if rv['r_value'] < 1000]
            if valid_r_values:
                best_result = min(valid_r_values, key=lambda x: x['r_value'])
                print(f"\n最佳R值: {best_result['r_value']} (进程 {best_result['process_id']})")
            else:
                print("\n所有进程均失败，无有效R值")
        else:
            print("\n未能从任何进程中提取到有效的R值")
            
        return {
            'execution_time': end_time - start_time,
            'results': final_results,
            'successful_r_values': successful_r_values,
            'best_r_value': best_result if valid_r_values else None,
            'temp_dir': base_temp_dir,  # 返回临时文件夹路径
            'all_successful': all_successful  # 返回整体成功状态
        }

    finally:
        # 清理临时文件夹 - 仅在全部成功时删除
        print("\n清理临时文件夹...")
        if all_successful:
            try:
                shutil.rmtree(base_temp_dir)
                print(f"已删除临时目录: {base_temp_dir}")
            except Exception as e:
                print(f"删除临时目录时出错: {e}")
        else:
            print(f"由于进程失败，保留临时文件夹用于调试: {base_temp_dir}")
            print("请检查以下文件夹中的日志和输出文件:")
            for folder_path in folder_paths:
                print(f"  - {folder_path}")

class LEED(SyntheticTestFunction):
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False
    _optimizers: List[Tuple[float, ...]]

    def __init__(self, dim: int = 3, noise_std: Optional[float] = None, negate: bool = False) -> None:
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(-0.5, 0.5) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
  
    def _evaluate_true(self, X, noise=0.1):
        loss = []
        parameters_file = "PARAMETERS"  # 请替换为你的参数文件路径
        shell_script_file = "leed.sh"     # 请替换为你的shell脚本路径
        VIBROCC = "VIBROCC"
        POSCAR = "POSCAR"
        EXPBEAMS = "EXPBEAMS.csv"
        # IVBEAMS = "IVBEAMS" 
        
        num_groups = X.shape[0] // 4  # 8//4=2组
        dim = X.shape[1]
        
        x_i = X.reshape(num_groups, 4, dim)
        
        for i in range(x_i.shape[0]):
            x = x_i[i]
        
            results = parallel_shell_execution(
                x,
                parameters_file=parameters_file,
                shell_script_file=shell_script_file,
                VIBROCC = VIBROCC,
                POSCAR =POSCAR,
                EXPBEAMS = EXPBEAMS, 
                num_processes=4
            )
            data = results['successful_r_values']
            r_values = [item['r_value'] for item in data]
            r_fac = torch.tensor(r_values, dtype=torch.float32).unsqueeze(1)
            write_tensor_to_file(x, r_fac, "temp.txt")
        
            loss.append(r_fac)
        
        return torch.cat(loss, dim=0).to(dtype=dtype, device=device)

low_tensor = torch.tensor(final_low_vector, dtype=dtype)
up_tensor = torch.tensor(final_up_vector, dtype=dtype)
# theta_low = torch.tensor([0], dtype=dtype)
# theta_up = torch.tensor([2], dtype=dtype)
# bounds = torch.empty(2, 13) 
# bounds[0, :] = torch.cat((low_tensor, theta_low))
# bounds[1, :] = torch.cat((up_tensor, theta_up))
# bounds[0, :] = torch.tensor(low_tensor,dtype=dtype)
# bounds[1, :] = torch.tensor(up_tensor,dtype=dtype)
print(low_tensor) 

fun = LEED(dim=13, negate=True).to(dtype=dtype, device=device) 
# low_tensor = torch.tensor(final_low_vector, dtype=dtype)
# up_tensor = torch.tensor(final_up_vector, dtype=dtype)
# theta_low = torch.tensor([0], dtype=dtype)
# theta_up = torch.tensor([2], dtype=dtype)
fun.bounds[0, :] = torch.tensor(low_tensor,dtype=dtype)
fun.bounds[1, :] = torch.tensor(up_tensor,dtype=dtype)
print(fun.bounds) 
dim_q= fun.dim
lb, ub = fun.bounds
n_init = 1000

def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))

@dataclass
class TurboState:
    dim_q: int
    batch_size: int
    length: float = 1.6
    length_min: float = 0.1**5
    length_max: float = 100
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim_q) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

batch_size = 16
state = TurboState(dim_q=dim_q, batch_size=batch_size)
print(state)
max_cholesky_size = float("inf")  # Always use Cholesky

def get_initial_points(dim_q, n_pts):
    sobol = SobolEngine(dimension=dim_q, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def restarts_BO_q():

    with open('04/'+'para_results.txt') as f:
        candi_lines = f.readlines()

    _X = []
    _Y = []
    for i in candi_lines:
        info = i.split()
        file_name = info[0]
        _x = np.array([float(num) for num in info[1].split('_') if num])
        _y = float(info[2])
                        
        _X.append(_x)
        _Y.append(_y)
        #candi_dict[file_name] = [x_term, y]

    tensor_x = normalize(torch.tensor(_X, dtype=dtype, device=device), fun.bounds)
    tensor_y = torch.tensor(_Y, dtype=dtype, device=device).unsqueeze(-1)

    return {'x':tensor_x, 'y':tensor_y}

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="qucb",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei","qucb")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim_q = X.shape[-1]
        sobol = SobolEngine(dim_q, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim_q, 1.0)
        mask = torch.rand(n_candidates, dim_q, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim_q - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim_q).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedim_qprovement(model, train_Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    elif acqf == "qucb":
        qucb = qUpperConfidenceBound(model, beta=1)
        X_next, acq_value = optimize_acqf(
            qucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

x_init = get_initial_points(dim_q, n_pts=128)
y_init = eval_objective(x_init)

X_tu, Y_tu, _, _ = read_and_process_data("output.txt")
X_in, Y_in,_,_ = read_and_process_data("temp.txt")

X_in = X_in.to(device=device,dtype=dtype)
Y_in = Y_in.to(device=device,dtype=dtype)
X_tu = X_tu.to(device=device,dtype=dtype)
Y_tu = Y_tu.to(device=device,dtype=dtype)

X_init = normalize(X_in, fun.bounds)
X_init = torch.clamp(normalize(X_in, fun.bounds),0.0,1.0)
X_turbo = torch.cat((X_tu, X_init), dim=0)
Y_turbo = torch.cat((Y_tu, Y_in), dim=0)

X_turbo = X_turbo.to(dtype=dtype)
Y_turbo = -Y_turbo.to(dtype=dtype)
Y_turbo = Y_turbo.unsqueeze(-1) 

state = TurboState(dim_q, batch_size=batch_size, best_value=max(Y_turbo).item())

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
N_CANDIDATES = min(5000, max(2000, 200 * dim_q)) if not SMOKE_TEST else 4

torch.manual_seed(0)

while not state.restart_triggered:  # Run until TuRBO converges
    # Fit a GP model
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(
            nu=2.5, ard_num_dims=dim_q, lengthscale_constraint=Interval(0.005, 4.0)
        )
    )
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = SingleTaskGP(
        X_turbo, train_Y, train_Yvar, covar_module=covar_module, likelihood=likelihood
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Do the fitting and acquisition function optimization inside the Cholesky context
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Fit the model
        fit_gpytorch_mll(mll)

        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            X=X_turbo,
            Y=train_Y,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf="qucb",
        )
        X_next = torch.clamp(X_next, 0.0, 1.0)
    Y_next =  eval_objective(X_next)

    # Update state
    state = update_state(state=state, Y_next=Y_next)

    # Append data
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
    # torch.cuda.empty_cache()

    # Print current status
    print(
        f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
    )
