import torch
import numpy as np

def write_tensor_to_file(params_tensor, results_tensor, file_path):
    n = params_tensor.size(0)
    if results_tensor.size(0) != n:
        raise ValueError(f"参数和结果样本数量不一致: 参数有{n}个样本，结果有{results_tensor.size(0)}个样本")
    
    if params_tensor.size(1) != 53:
        raise ValueError(f"参数维度错误: 应为53列，实际为{params_tensor.size(1)}列")

    if results_tensor.dim() == 1:
        results_tensor = results_tensor.view(-1, 1)

    with open(file_path, "a") as f:
        for i in range(n):
            params_line = params_tensor[i].tolist()    
            params_str = "_".join([f"{x:.6f}" for x in params_line])
            result_value = results_tensor[i].item()
            line = f"{params_str} {result_value:.6f}\n"    
            f.write(line)
            
def load_tensor_from_file(file_path):
    params_list = []
    results_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)  

            if len(parts) < 2:
                raise ValueError(f"行格式错误: {line}")                
            param_str, result_str = parts
            param_values = [float(x) for x in param_str.split('_')]

            result_values = [float(x) for x in result_str.split()]
         
            params_list.append(param_values)
            results_list.append(result_values)

    params_tensor = torch.tensor(params_list, dtype=torch.float32)
    results_tensor = torch.tensor(results_list, dtype=torch.float32)
    
    return params_tensor, results_tensor

def read_and_process_data(filename):
   
    all_X = []
    all_Y = []
    
    # Read the file
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    # Split by underscore and space
                    parts = line.replace('_', ' ').split()
                    
                    # Convert to float
                    values = [float(x) for x in parts]
                    
                    # Last value is Y, rest are X
                    X = values[:-1]
                    Y = values[-1]
                    
                    all_X.append(X)
                    all_Y.append(Y)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None, None
    
    # Convert to numpy arrays
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)
    
    # Split based on Y values
    low_mask = all_Y < 2
    high_mask = all_Y >= 2
    
    X_low = all_X[low_mask]
    Y_low = all_Y[low_mask]
    X_high = all_X[high_mask]
    Y_high = all_Y[high_mask]
    
    # Convert to PyTorch tensors
    X_low_tensor = torch.tensor(X_low, dtype=torch.float32)
    Y_low_tensor = torch.tensor(Y_low, dtype=torch.float32)
    X_high_tensor = torch.tensor(X_high, dtype=torch.float32)
    Y_high_tensor = torch.tensor(Y_high, dtype=torch.float32)
    
    return X_low_tensor, Y_low_tensor, X_high_tensor, Y_high_tensor