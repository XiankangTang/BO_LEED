import os
import sys,re
import tempfile
import numpy as np
import subprocess
import multiprocessing
from copy import deepcopy
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
from iocode.gp import train_gp
from iocode.turbo_1 import Turbo1
from iocode.utils import from_unit_cube, latin_hypercube, to_unit_cube

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# os.system('export OMP_NUM_THREADS=48')
# os.system('ulimit -v 48000000')
root_path = os.getcwd()+'/'

def frac_to_cart(frac_coords, lattice):
    return np.dot(frac_coords, lattice)

def cart_to_frac(cart_coords, lattice):
    return np.dot(cart_coords, np.linalg.inv(lattice))

lattice, atom_types, atom_counts, coords, coord_type = read_POSCAR("pos/POSCAR")
if coord_type == 'c':
    cart_coords = coords
    frac_coords = cart_to_frac(cart_coords, lattice)
else:
    frac_coords = coords
    cart_coords = frac_to_cart(frac_coords, lattice)

# Select atoms with fractional z > 0.25
mask = frac_coords[:, 2] > 0.25
selected_frac = frac_coords[mask]
selected_cart = cart_coords[mask]

print(f"Number of selected atoms: {len(selected_cart)}")

delta = 0.3
lower_cart = selected_cart - delta
upper_cart = selected_cart + delta

lower_frac = cart_to_frac(lower_cart, lattice)
upper_frac = cart_to_frac(upper_cart, lattice)

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
target_keys = ['O_def', 'Fe_def']
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
def execute_in_folder(folder_path, shell_script_name):

    try:
       
        original_cwd = os.getcwd()
        os.chdir(folder_path)
       
        shell_script_path = os.path.join(folder_path, shell_script_name)
        os.chmod(shell_script_path, 0o755)
    
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
            'stderr': 'EXECUTION_TIMEOUT',
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
    x5 = Num_x[8]
    y5 = Num_x[9]
    x6 = Num_x[10]
    y6 = Num_x[11]
    x7 = Num_x[12]
    y7 = Num_x[13]
    x8 = Num_x[14]
    y8 = Num_x[15]
    x9 = Num_x[16]
    y9 = Num_x[17]
    x10 = Num_x[18]
    y10 = Num_x[19]
    x11 = Num_x[20]
    y11 = Num_x[21]
    x12 = Num_x[22]
    y12 = Num_x[23]
    x13 = Num_x[24]
    y13 = Num_x[25]
    x14 = Num_x[26]
    y14 = Num_x[27]
    x15 = Num_x[28]
    y15 = Num_x[29]
    x16 = Num_x[30]
    y16 = Num_x[31]
    x17 = Num_x[32]
    y17 = Num_x[33]
    x18 = Num_x[34]
    y18 = Num_x[35]
    x19 = Num_x[36]
    y19 = Num_x[37]
    x20 = Num_x[38]
    y20 = Num_x[39]
    z1 = Num_x[40]
    z2 = Num_x[41]
    z3 = Num_x[42]
    z4 = Num_x[43]
    z5 = Num_x[44]
    z6 = Num_x[45]
    z7 = Num_x[46]
    z8 = Num_x[47]
    z9 = Num_x[48]
    z10 = Num_x[49]
    O = Num_x[50]
    Fe = Num_x[51]
    theta = Num_x[52]
    data = read_poscar(filepath)
    modifications = {
        0: [x1, y1, z1],  
        1: [x2, y2, z1],   
        2: [x3, y3, z2],  
        3: [x4, y4, z2],   
        4: [x5, y5, z3],  
        5: [x6, y6, z3], 
        6: [x7, y7, z4],   
        7: [x8, y8, z4],  
        16: [x9, y9, z5], 
        17: [x10, y10, z5],
        18: [x11, y11, z6],
        19: [x12, y12, z6],
        20: [x13, y13, z7],
        21: [x14, y14, z7],
        22: [x15, y15, z8],
        23: [x16, y16, z8],
        24: [x17, y17, z9],
        25: [x18, y18, z9],
        26: [x19, y19, z10],
        27: [x20, y20, z10],
    }
    modified_data = batch_modify_poscar_data(data, modifications)
    write_poscar(modified_data, poscar)
    read_and_update_config(vibrocc,updates = {"Fe_surf":Fe,"O_surf":O})
    beam = {'BEAM_INCIDENCE': 'THETA 0.0, PHI 90.0000'}
    updated_beam = update_theta(beam, theta)
    read_and_update_config(paras,updates = updated_beam)
def parallel_shell_execution(X, parameters_file, shell_script_file, VIBROCC, POSCAR, EXPBEAMS, IVBEAMS, num_processes=4):
    
    param_subsets = torch.split(X, 1, dim=0)

    filepath = "/work/scratch/xt17pyku/LEEDtest/Fe04/pos/POSCAR"
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
        IVBEAMS_name = os.path.basename(IVBEAMS)

        for i, folder_path in enumerate(folder_paths):
            shutil.copy2(parameters_file, os.path.join(folder_path, parameters_name))
            shutil.copy2(shell_script_file, os.path.join(folder_path, shell_script_name))
            shutil.copy2(POSCAR, os.path.join(folder_path, POSCAR_name))
            shutil.copy2(VIBROCC, os.path.join(folder_path, VIBROCC_name))
            shutil.copy2(EXPBEAMS, os.path.join(folder_path, EXPBEAMS_name))
            shutil.copy2(IVBEAMS, os.path.join(folder_path, IVBEAMS_name))
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
        any_error = False  
        for i, result in enumerate(results, 1):

            if result['stdout']:
                print(f"OUTPUT:\n{result['stdout'][:500]}...")
            if result['stderr']:
                print(f"ERROR:\n{result['stderr'][:500]}...")
            
            log_result = None
            process_has_error = False 
            
            if result['success']:
                newest_log = get_newest_log_file(result['folder'])
                if newest_log:
                    log_result = extract_r_value_from_log(newest_log)
                    if log_result['success']:
           
                        if log_result['r_float'] == 0:
   
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
                       
                        log_result = {
                            'log_file': newest_log,
                            'r_value': "1000 ",
                            'r_float': 1000.0,
                            'success': False,
                            'error': 'EXTRACTION_FAILED'
                        }
                        process_has_error = True
                else:

                    log_result = {
                        'log_file': None,
                        'r_value': "1000 ",
                        'r_float': 1000.0,
                        'success': False,
                        'error': 'NO_LOG'
                    }
                    process_has_error = True
            else:
                
                log_result = {
                    'log_file': None,
                    'r_value': "1000",
                    'r_float': 1000.0,
                    'success': False,
                    'error': 'EXECUTION_FAILED'
                }
                process_has_error = True

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
            
            if not process_has_error:
                display.clear_output(wait=True)
        
        successful_r_values = []
        for result in final_results:
            if result['log_analysis']['success'] and result['log_analysis']['r_float'] != 0:
                successful_r_values.append({
                    'process_id': result['process_id'],
                    'r_value': result['log_analysis']['r_float'],
                    'log_file': result['log_analysis']['log_file']
                })
            else:
                successful_r_values.append({
                    'process_id': result['process_id'],
                    'r_value': 1000.0,
                    'log_file': None
                })
 
        if successful_r_values:
            print(f" {len([rv for rv in successful_r_values if rv['r_value'] < 1000])} R:")
            for rv in successful_r_values:
                if rv['r_value'] < 1000:
                    print(f"   {rv['process_id']}: R = {rv['r_value']}")
                else:
                    print(f"   {rv['process_id']}: R = 1000 (Failed)")
            
           
            valid_r_values = [rv for rv in successful_r_values if rv['r_value'] < 1000]
            if valid_r_values:
                best_result = min(valid_r_values, key=lambda x: x['r_value'])
                print(f"\nR: {best_result['r_value']} (process {best_result['process_id']})")
            else:
                print("\nAll falied，No R")
        else:
            print("\nNo R")
            
        return {
            'execution_time': end_time - start_time,
            'results': final_results,
            'successful_r_values': successful_r_values,
            'best_r_value': best_result if valid_r_values else None,
            'temp_dir': base_temp_dir,  
            'all_successful': all_successful  
        }

    finally:
           
        if all_successful:
            try:
                shutil.rmtree(base_temp_dir)
               
            except Exception as e:
                print(f{e})
        else:
          
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
        parameters_file = "PARAMETERS"  
        shell_script_file = "leed.sh"     
        VIBROCC = "VIBROCC"
        POSCAR = "POSCAR"
        EXPBEAMS = "EXPBEAMS.csv"
        IVBEAMS = "IVBEAMS" 
        
        num_groups = X.shape[0] // 4  
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
                IVBEAMS = IVBEAMS,
                num_processes=4
            )
            data = results['successful_r_values']
            r_values = [item['r_value'] for item in data]
            r_fac = torch.tensor(r_values, dtype=torch.float32).unsqueeze(1)
            write_tensor_to_file(x, r_fac, "temp.txt")
        
            loss.append(r_fac)
            # print(loss)
        
        return torch.cat(loss, dim=0).to(dtype=dtype, device=device)
    
fun = LEED(dim=53, negate=True).to(dtype=dtype, device=device) 
low_tensor = torch.tensor(final_low_vector, dtype=dtype)
up_tensor = torch.tensor(final_up_vector, dtype=dtype)
theta_low = torch.tensor([0], dtype=dtype)
theta_up = torch.tensor([2], dtype=dtype)
fun.bounds[0, :] = torch.cat((low_tensor, theta_low))
fun.bounds[1, :] = torch.cat((up_tensor, theta_up))
print(fun.bounds) 
dim_q= fun.dim
lb, ub = fun.bounds
lb = lb.cpu().numpy()
ub = ub.cpu().numpy()

def to_unit(x, lb, ub):
    """
    Normalize data to unit cube [0, 1]^d
    x: input array (n_samples, n_features)
    lb: lower bounds for each feature
    ub: upper bounds for each feature
    """
    return (x - lb) / (ub - lb)
# Read data from temp.txt
with open('temp.txt', 'r') as f:
    lines = f.readlines()

data = []
all_x = []

# Parse each line and collect x values
for line in lines:
    line = line.strip()
    if not line:  # Skip empty lines
        continue
        
    parts = line.split()
    if len(parts) < 2:
        continue  # Skip invalid lines
        
    x_str, y_str = parts[0], parts[-1]
    x_list = list(map(float, x_str.split('_')))
    y = float(y_str)
    data.append((x_list, y))
    all_x.append(x_list)

# Convert to numpy array
all_x = np.array(all_x)

# Normalize x values using your to_unit_cube function
output_lines = []
for x_list, y in data:
    # x_normalized = to_unit(np.array(x_list), lb, ub)
    x_normalized = np.array(x_list)
    # Format normalized x values (remove trailing zeros)
    x_str = '_'.join(f'{val:.6f}'.rstrip('0').rstrip('.') for val in x_normalized)
    output_lines.append(f"{x_str} {y}\n")

# Write normalized data to out.txt
with open('output.txt', 'a') as f:
    f.writelines(output_lines)

# Clean temp.txt by truncating
open('temp.txt', 'w').close()

print("Data processing completed!")
print(f"Processed {len(data)} records")
print(f"Lower bounds: {lb}")
print(f"Upper bounds: {ub}")

class TurboM(Turbo1):
    """The TuRBO-m algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points *FOR EACH TRUST REGION* (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    n_trust_regions : Number of trust regions
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo5 = TurboM(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
        turbo5.optimize()  # Run optimization
        X, fX = turbo5.X, turbo5.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):
        self.n_trust_regions = n_trust_regions
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )

        self.succtol = 3
        self.failtol = max(5, self.dim)

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)
    
    def eval_objective(self, x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        x = torch.tensor(x, device=self.device, dtype=self.dtype)
        return self.f(unnormalize(x, self.f.bounds))

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next

    def optimize(self):
        """Run the full optimization process."""
        # Create initial points for each TR
        for i in range(self.n_trust_regions):
            X_init = latin_hypercube(self.n_init, self.dim)

            y_init = self.eval_objective(X_init)
         
            X_tu, Y_tu, _, _ = read_and_process_data("output.txt")
            X_in, Y_in,_,_ = read_and_process_data("temp.txt")

            X_in = X_in.to(device=device,dtype=dtype)
            Y_in = Y_in.to(device=device,dtype=dtype)
            X_tu = X_tu.to(device=device,dtype=dtype)
            Y_tu = Y_tu.to(device=device,dtype=dtype)

            X_init = normalize(X_in, fun.bounds)
            X_tun = normalize(X_tu, fun.bounds)
            X_init = torch.clamp(normalize(X_in, fun.bounds),0.0,1.0)
            X_turbo = torch.cat((X_tun, X_init), dim=0)
            Y_turbo = torch.cat((Y_tu, Y_in), dim=0)

            X_turbo = X_turbo.to(dtype=dtype)
            Y_turbo = Y_turbo.to(dtype=dtype)  ## remove the minus sign
            Y_turbo = Y_turbo.unsqueeze(-1) 
            # X_init = from_unit_cube(X_init, self.lb, self.ub)
            # fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            # self.X = np.vstack((self.X, X_init))
            # self.fX = np.vstack((self.fX, fX_init))
            self.X = X_turbo.cpu().numpy()
            self.fX = Y_turbo.cpu().numpy()
            total_points = X_turbo.shape[0]  
            
            # self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            if i == 0:
                
                self._idx = i * np.ones((total_points, 1), dtype=int)
            else:
            
                existing_points = self._idx.shape[0]
                new_points = total_points - existing_points
                if new_points > 0:
                    new_idx = i * np.ones((new_points, 1), dtype=int)
                    self._idx = np.vstack((self._idx, new_idx))
              
                elif new_points < 0:
                    self._idx = self._idx[:total_points, :]
            self.n_evals += self.n_init

            if self.verbose:
                # fbest = fX_init.min()
                fbest = Y_turbo.min()
                print(f"TR-{i} starting from: {fbest:.4}")
                sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # Generate candidates from each TR
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X_c = deepcopy(self.X[idx, :])
                # X = to_unit_cube(X, self.lb, self.ub)
                X = np.clip(X_c,0.0,1.0)
                print(X.min(),X.max())

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())
            
                # Don't retrain the model if the training data hasn't changed
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                    X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            # X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            # fX_next = np.array([[self.f(x)] for x in X_next])
            fX_ne = self.eval_objective(torch.tensor(X_next, dtype=dtype, device=device)).cpu().numpy()
            fX_next = -fX_ne
            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                print("restart check!!!!!!")
                if self.length[i] < self.length_min:  # Restart trust region if converged
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    # X_init = from_unit_cube(X_init, self.lb, self.ub)
                    # fX_init = np.array([[self.f(x)] for x in X_init])
                    fX_init_t = self.eval_objective(X_init)
                    fX_init = fX_init_t.cpu().numpy()

                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init

turbo_m = TurboM(
    f=fun,  # Handle to objective function
    lb=lb,  # Numpy array specifying lower bounds
    ub=ub,  # Numpy array specifying upper bounds
    n_init=16,  # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=5000,  # Maximum number of evaluations
    n_trust_regions=4,  # Number of trust regions
    batch_size=4,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo_m.optimize()