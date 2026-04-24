import os, re
import glob

def get_newest_log_file(folder_path):

    log_pattern = os.path.join(folder_path, "viperleed-calc-*.log")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        return None
    
    # 根据文件修改时间排序，获取最新的文件
    newest_log = max(log_files, key=os.path.getmtime)
    return newest_log

def extract_r_value_from_log(log_file_path):
    result = {
        'log_file': log_file_path,
        'r_value': None,
        'r_float': None,
        'success': False,
        'error': None
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取R值
        match = re.search(r"Final R \(refcalc\):\s*([0-9.]+)", content)
        
        if match:
            r_value = match.group(1)
            result['r_value'] = r_value
            result['success'] = True
            
            try:
                r_float = float(r_value)
                result['r_float'] = r_float
            except ValueError:
                result['error'] = "无法转换R值为浮点数"
        else:
            result['error'] = "未找到 Final R (refcalc) 值"
            
    except Exception as e:
        result['error'] = f"读取日志文件时出错: {str(e)}"
    
    return result

def write_parameters_file(file_path, params):
    """Write parameters from a tensor row to a file"""
    with open(file_path, 'w') as f:
        for value in params:
            f.write(f"{value}\n")