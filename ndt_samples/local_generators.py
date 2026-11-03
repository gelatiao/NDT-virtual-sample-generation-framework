# Extracted from local TC scripts without importing their training side effects.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def mlp_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数 - 简化版本"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_ranges = np.ptp(X_array, axis=0)  # 使用数据范围代替标准差
    
    for _ in range(n_virtual):
        # 随机选择基础样本
        base_indices = np.random.choice(n_samples, 3, replace=False)
        weights = np.random.dirichlet(np.ones(3))
        
        # 生成新样本
        base_samples = X_array[base_indices]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        
        # 添加噪声
        noise = np.random.uniform(-noise_ratio, noise_ratio, n_features) * feature_ranges
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0),
                               X_array.max(axis=0))
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        base_target = np.average(y_array[base_indices], weights=weights)
        target_noise = np.random.uniform(-noise_ratio, noise_ratio) * np.ptp(y_array)
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array),
                               max(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets

def rf_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_means = np.mean(X_array, axis=0)
    feature_stds = np.std(X_array, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)
    
    # 特征相关性约束
    corr_matrix = np.corrcoef(X_array.T)
    
    # 计算样本之间的距离矩阵
    from sklearn.metrics.pairwise import euclidean_distances
    dist_matrix = euclidean_distances(X_array)
    
    # 计算每个样本的密度
    density = np.sum(1 / (1 + dist_matrix), axis=1)
    
    for _ in range(n_virtual):
        # 优先选择低密度区域的样本
        weights = 1 / (density + 1e-6)
        weights /= np.sum(weights)
        base_idx = np.random.choice(n_samples, p=weights)
        
        # 选择距离适中的样本对
        distances = dist_matrix[base_idx]
        valid_indices = np.where((distances > np.percentile(distances, 30)) & 
                               (distances < np.percentile(distances, 70)))[0]
        if len(valid_indices) >= 2:
            other_indices = np.random.choice(valid_indices, 2, replace=False)
        else:
            other_indices = np.random.choice(range(n_samples), 2, replace=False)
        
        # 使用自适应权重进行插值
        weights = np.random.dirichlet(np.ones(3) * 2)  # 增加alpha参数使权重更均匀
        base_samples = X_array[np.append(base_idx, other_indices)]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        base_target = np.average(y_array[np.append(base_idx, other_indices)], weights=weights)
        
        # 自适应噪声
        noise = np.zeros(n_features)
        for j in range(n_features):
            # 考虑特征相关性的噪声生成
            related_features = np.where(np.abs(corr_matrix[j]) > 0.5)[0]
            if len(related_features) > 1:
                # 相关特征的联合噪声
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
                for k in related_features:
                    if k != j:
                        noise[j] += 0.5 * corr_matrix[j,k] * noise[k]  # 减小相关性影响
            else:
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
            
            # 限制噪声范围
            noise[j] = np.clip(noise[j], 
                             -1.5 * feature_stds[j],  # 减小噪声范围
                             1.5 * feature_stds[j])
        
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0) - 0.1 * feature_stds,
                               X_array.max(axis=0) + 0.1 * feature_stds)
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        target_noise = np.random.normal(0, noise_ratio * np.std(y_array))
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array) - 0.1 * np.std(y_array),
                               max(y_array) + 0.1 * np.std(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets

def svr_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_means = np.mean(X_array, axis=0)
    feature_stds = np.std(X_array, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)
    
    # 特征相关性约束
    corr_matrix = np.corrcoef(X_array.T)
    
    # 计算样本之间的距离矩阵
    dist_matrix = euclidean_distances(X_array)
    
    # 计算每个样本的密度
    density = np.sum(1 / (1 + dist_matrix), axis=1)
    
    for _ in range(n_virtual):
        # 优先选择低密度区域的样本
        weights = 1 / (density + 1e-6)
        weights /= np.sum(weights)
        base_idx = np.random.choice(n_samples, p=weights)
        
        # 选择距离适中的样本对
        distances = dist_matrix[base_idx]
        valid_indices = np.where((distances > np.percentile(distances, 30)) & 
                               (distances < np.percentile(distances, 70)))[0]
        if len(valid_indices) >= 2:
            other_indices = np.random.choice(valid_indices, 2, replace=False)
        else:
            other_indices = np.random.choice(range(n_samples), 2, replace=False)
        
        # 使用自适应权重进行插值
        weights = np.random.dirichlet(np.ones(3) * 2)
        base_samples = X_array[np.append(base_idx, other_indices)]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        base_target = np.average(y_array[np.append(base_idx, other_indices)], weights=weights)
        
        # 自适应噪声
        noise = np.zeros(n_features)
        for j in range(n_features):
            # 考虑特征相关性的噪声生成
            related_features = np.where(np.abs(corr_matrix[j]) > 0.5)[0]
            if len(related_features) > 1:
                # 相关特征的联合噪声
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
                for k in related_features:
                    if k != j:
                        noise[j] += 0.5 * corr_matrix[j,k] * noise[k]
            else:
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
            
            # 限制噪声范围
            noise[j] = np.clip(noise[j], 
                             -1.5 * feature_stds[j],
                             1.5 * feature_stds[j])
        
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0) - 0.1 * feature_stds,
                               X_array.max(axis=0) + 0.1 * feature_stds)
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        target_noise = np.random.normal(0, noise_ratio * np.std(y_array))
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array) - 0.1 * np.std(y_array),
                               max(y_array) + 0.1 * np.std(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets

def gbdt_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_means = np.mean(X_array, axis=0)
    feature_stds = np.std(X_array, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)
    
    # 特征相关性约束
    corr_matrix = np.corrcoef(X_array.T)
    
    # 计算样本之间的距离矩阵
    dist_matrix = euclidean_distances(X_array)
    
    # 计算每个样本的密度
    density = np.sum(1 / (1 + dist_matrix), axis=1)
    
    for _ in range(n_virtual):
        # 优先选择低密度区域的样本
        weights = 1 / (density + 1e-6)
        weights /= np.sum(weights)
        base_idx = np.random.choice(n_samples, p=weights)
        
        # 选择距离适中的样本对
        distances = dist_matrix[base_idx]
        valid_indices = np.where((distances > np.percentile(distances, 40)) & 
                               (distances < np.percentile(distances, 60)))[0]  # 缩小有效样本范围
        if len(valid_indices) >= 2:
            other_indices = np.random.choice(valid_indices, 2, replace=False)
        else:
            other_indices = np.random.choice(range(n_samples), 2, replace=False)
        
        # 使用自适应权重进行插值
        weights = np.random.dirichlet(np.ones(3) * 3)  # 增加权重集中度
        base_samples = X_array[np.append(base_idx, other_indices)]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        base_target = np.average(y_array[np.append(base_idx, other_indices)], weights=weights)
        
        # 自适应噪声
        noise = np.zeros(n_features)
        for j in range(n_features):
            # 考虑特征相关性的噪声生成
            related_features = np.where(np.abs(corr_matrix[j]) > 0.5)[0]
            if len(related_features) > 1:
                # 相关特征的联合噪声
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j] * 0.8)  # 降低噪声水平
                for k in related_features:
                    if k != j:
                        noise[j] += 0.3 * corr_matrix[j,k] * noise[k]  # 降低相关性影响
            else:
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j] * 0.8)
            
            # 限制噪声范围
            noise[j] = np.clip(noise[j], 
                             -1.0 * feature_stds[j],  # 收紧噪声范围
                             1.0 * feature_stds[j])
        
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0) - 0.05 * feature_stds,  # 收紧范围
                               X_array.max(axis=0) + 0.05 * feature_stds)
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        target_noise = np.random.normal(0, noise_ratio * np.std(y_array) * 0.8)  # 降低目标值噪声
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array) - 0.05 * np.std(y_array),  # 收紧范围
                               max(y_array) + 0.05 * np.std(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets

def catboost_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_means = np.mean(X_array, axis=0)
    feature_stds = np.std(X_array, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)
    
    # 特征相关性约束
    corr_matrix = np.corrcoef(X_array.T)
    
    # 计算样本之间的距离矩阵
    dist_matrix = euclidean_distances(X_array)
    
    # 计算每个样本的密度
    density = np.sum(1 / (1 + dist_matrix), axis=1)
    
    for _ in range(n_virtual):
        # 优先选择低密度区域的样本
        weights = 1 / (density + 1e-6)
        weights /= np.sum(weights)
        base_idx = np.random.choice(n_samples, p=weights)
        
        # 选择距离适中的样本对
        distances = dist_matrix[base_idx]
        valid_indices = np.where((distances > np.percentile(distances, 30)) & 
                               (distances < np.percentile(distances, 70)))[0]
        if len(valid_indices) >= 2:
            other_indices = np.random.choice(valid_indices, 2, replace=False)
        else:
            other_indices = np.random.choice(range(n_samples), 2, replace=False)
        
        # 使用自适应权重进行插值
        weights = np.random.dirichlet(np.ones(3) * 2)
        base_samples = X_array[np.append(base_idx, other_indices)]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        base_target = np.average(y_array[np.append(base_idx, other_indices)], weights=weights)
        
        # 自适应噪声
        noise = np.zeros(n_features)
        for j in range(n_features):
            # 考虑特征相关性的噪声生成
            related_features = np.where(np.abs(corr_matrix[j]) > 0.5)[0]
            if len(related_features) > 1:
                # 相关特征的联合噪声
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
                for k in related_features:
                    if k != j:
                        noise[j] += 0.5 * corr_matrix[j,k] * noise[k]
            else:
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
            
            # 限制噪声范围
            noise[j] = np.clip(noise[j], 
                             -1.5 * feature_stds[j],
                             1.5 * feature_stds[j])
        
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0) - 0.1 * feature_stds,
                               X_array.max(axis=0) + 0.1 * feature_stds)
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        target_noise = np.random.normal(0, noise_ratio * np.std(y_array))
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array) - 0.1 * np.std(y_array),
                               max(y_array) + 0.1 * np.std(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets

def xgboost_samples(X, y, noise_ratio=0.01, n_virtual=100):
    """改进的虚拟样本生成函数"""
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns if hasattr(X, 'columns') else None
    n_samples, n_features = X_array.shape
    virtual_samples = []
    virtual_targets = []
    
    # 计算特征统计信息
    feature_means = np.mean(X_array, axis=0)
    feature_stds = np.std(X_array, axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)
    
    # 特征相关性约束
    corr_matrix = np.corrcoef(X_array.T)
    
    # 计算样本之间的距离矩阵
    dist_matrix = euclidean_distances(X_array)
    
    # 计算每个样本的密度
    density = np.sum(1 / (1 + dist_matrix), axis=1)
    
    for _ in range(n_virtual):
        # 优先选择低密度区域的样本
        weights = 1 / (density + 1e-6)
        weights /= np.sum(weights)
        base_idx = np.random.choice(n_samples, p=weights)
        
        # 选择距离适中的样本对
        distances = dist_matrix[base_idx]
        valid_indices = np.where((distances > np.percentile(distances, 30)) & 
                               (distances < np.percentile(distances, 70)))[0]
        if len(valid_indices) >= 2:
            other_indices = np.random.choice(valid_indices, 2, replace=False)
        else:
            other_indices = np.random.choice(range(n_samples), 2, replace=False)
        
        # 使用自适应权重进行插值
        weights = np.random.dirichlet(np.ones(3) * 2)
        base_samples = X_array[np.append(base_idx, other_indices)]
        base_sample = np.average(base_samples, weights=weights, axis=0)
        base_target = np.average(y_array[np.append(base_idx, other_indices)], weights=weights)
        
        # 自适应噪声
        noise = np.zeros(n_features)
        for j in range(n_features):
            # 考虑特征相关性的噪声生成
            related_features = np.where(np.abs(corr_matrix[j]) > 0.5)[0]
            if len(related_features) > 1:
                # 相关特征的联合噪声
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
                for k in related_features:
                    if k != j:
                        noise[j] += 0.5 * corr_matrix[j,k] * noise[k]
            else:
                noise[j] = np.random.normal(0, noise_ratio * feature_stds[j])
            
            # 限制噪声范围
            noise[j] = np.clip(noise[j], 
                             -1.5 * feature_stds[j],
                             1.5 * feature_stds[j])
        
        virtual_sample = base_sample + noise
        
        # 确保生成的样本在合理范围内
        virtual_sample = np.clip(virtual_sample,
                               X_array.min(axis=0) - 0.1 * feature_stds,
                               X_array.max(axis=0) + 0.1 * feature_stds)
        
        virtual_samples.append(virtual_sample)
        
        # 为虚拟样本生成目标值
        target_noise = np.random.normal(0, noise_ratio * np.std(y_array))
        virtual_target = base_target + target_noise
        virtual_target = np.clip(virtual_target,
                               min(y_array) - 0.1 * np.std(y_array),
                               max(y_array) + 0.1 * np.std(y_array))
        virtual_targets.append(virtual_target)
    
    virtual_samples = np.array(virtual_samples)
    virtual_targets = np.array(virtual_targets)
    
    if feature_names is not None:
        return pd.DataFrame(virtual_samples, columns=feature_names), virtual_targets
    return virtual_samples, virtual_targets