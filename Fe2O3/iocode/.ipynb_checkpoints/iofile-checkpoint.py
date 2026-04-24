import os,re
import numpy as np

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

def read_poscar(file_path):
    """读取 POSCAR 文件并将数据存储在字典中"""
    poscar_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    poscar_data['title'] = lines[0].strip()
    poscar_data['scaling_factor'] = float(lines[1].strip())
    poscar_data['lattice_vectors'] = [list(map(float, lines[i].strip().split())) for i in range(2, 5)]
    poscar_data['atom_types'] = lines[5].strip().split()
    poscar_data['atom_counts'] = list(map(int, lines[6].strip().split()))
    line_8 = lines[7].strip().lower()
    
    if line_8.startswith('selective dynamics'):
        poscar_data['selective_dynamics'] = True
        poscar_data['coordinate_type'] = lines[8].strip().capitalize()
        start_index = 9
    else:
        poscar_data['selective_dynamics'] = False
        poscar_data['coordinate_type'] = line_8.capitalize()
        start_index = 8

    atom_coordinates = []
    selective_flags = []
    for i in range(sum(poscar_data['atom_counts'])):
        line_data = lines[start_index + i].strip().split()
        atom_coordinates.append(list(map(float, line_data[:3])))
        if poscar_data['selective_dynamics']:
            selective_flags.append(line_data[3:])

    poscar_data['atom_coordinates'] = atom_coordinates
    if poscar_data['selective_dynamics']:
        poscar_data['selective_flags'] = selective_flags

    return poscar_data

def write_poscar(poscar_data, output_path):
    """将字典中的数据写回到 POSCAR 文件，并优化格式"""
    with open(output_path, 'w') as file:
        # 标题行（无缩进）
        file.write(f"{poscar_data['title']}\n")
        # 缩放因子（缩进2个空格）
        file.write(f"  {poscar_data['scaling_factor']}\n")
        
        # 晶格矢量（每行缩进4个空格）
        for vector in poscar_data['lattice_vectors']:
            file.write("    " + "    ".join(f"{v:.10f}" for v in vector) + "\n")
        
        # 原子类型（无缩进）
        file.write("    " +"   ".join(poscar_data['atom_types']) + "\n")
        
        # 原子数量（无缩进）
        file.write("    " +"   ".join(map(str, poscar_data['atom_counts'])) + "\n")
        
        # 选择性动力学标记（如有）
        if poscar_data['selective_dynamics']:
            file.write("Selective dynamics\n")
        
        # 坐标类型（无缩进）
        file.write(f"{poscar_data['coordinate_type']}\n")
        
        # 原子坐标（每行缩进2个空格）
        for i, coord in enumerate(poscar_data['atom_coordinates']):
            # 写入坐标（缩进2个空格）
            coord_str = "  " + "  ".join(f"{c:.10f}" for c in coord)
            
            # 添加选择性动力学标记（如有）
            if poscar_data.get('selective_flags'):
                flags = " ".join(poscar_data['selective_flags'][i])
                coord_str += " " + flags
            
            file.write(coord_str + "\n")
            
def batch_modify_poscar_data(data, modifications):
    """
    批量修改多个原子坐标
    
    参数:
    data - 原始数据字典
    modifications - 字典格式 {原子索引: 新坐标}
    
    返回:
    修改后的数据字典
    """
    modified_data = data.copy()
    for index, new_coord in modifications.items():
        if 0 <= index < len(modified_data['atom_coordinates']):
            if isinstance(new_coord, (list, tuple)) and len(new_coord) == 3:
                modified_data['atom_coordinates'][index] = list(new_coord)
            else:
                raise ValueError(f"原子 {index} 的坐标格式无效")
        else:
            raise IndexError(f"原子索引 {index} 超出范围")
    return modified_data

def remove_tensors_folder(parent_dir):
    
    tensors_path = os.path.join(parent_dir, "Tensors")
    if os.path.exists(tensors_path) and os.path.isdir(tensors_path):
        try:
            # 递归删除整个文件夹
            shutil.rmtree(tensors_path)
            print(f"成功删除文件夹: {tensors_path}")
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False
    else:
        print(f"未找到文件夹: {tensors_path}")
        return False
def update_theta(beam_dict, theta):
    """
    Update the THETA value in the BEAM_INCIDENCE key.
    """
    original = beam_dict['BEAM_INCIDENCE']

    # Replace only the THETA value
    updated = re.sub(r'THETA\s+[\d\.Ee+-]+', f'THETA {theta:.4f}', original)

    # Update the dictionary
    beam_dict['BEAM_INCIDENCE'] = updated
    return beam_dict

def read_and_update_config(filename, updates=None, new_filename=None):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    new_lines = lines.copy()
    params = {}
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('!') or not stripped_line:
            continue
        if '=' in line:
            parts = line.split('=', 1)
            key = parts[0].strip()
            value_str = parts[1].strip()
            if key in params:
                print(f"Warning: duplicate key '{key}' at line {i}, overwriting previous occurrence.")
            params[key] = (i, value_str)
    
    if updates:
        for key, new_value in updates.items():
            if key in params:
                line_idx, _ = params[key]
                new_line_str = f"{key} = {new_value}"
                if lines[line_idx].endswith('\n'):
                    new_line_str += '\n'
                new_lines[line_idx] = new_line_str
            else:
                print(f"Warning: key '{key}' not found. Skipping update.")
    
    out_file = new_filename if new_filename else filename
    with open(out_file, 'w') as f:
        f.writelines(new_lines)
    
    param_dict = {key: value for key, (_, value) in params.items()}
    return param_dict
