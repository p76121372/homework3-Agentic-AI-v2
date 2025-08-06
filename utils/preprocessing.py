import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any

class DataPreprocessor:
    """
    數據預處理器，用於處理感測器數據
    支持訓練模式和評估模式
    """
    
    def __init__(self):
        self.stats_file = 'model_weight/preprocessing_stats.pkl'
        self.stats = None
    
    def save_stats(self, stats: Dict[str, Any]) -> None:
        """
        保存統計信息到文件
        
        Args:
            stats (Dict[str, Any]): 統計信息字典
        """
        try:
            with open(self.stats_file, 'wb') as f:
                pickle.dump(stats, f)
            print(f"✅ 統計信息已保存到 {self.stats_file}")
        except Exception as e:
            print(f"❌ 保存統計信息失敗: {e}")
    
    def load_stats(self) -> Dict[str, Any]:
        """
        從文件加載統計信息
        
        Returns:
            Dict[str, Any]: 統計信息字典
        """
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'rb') as f:
                    stats = pickle.load(f)
                #print(f"✅ 統計信息已從 {self.stats_file} 加載")
                return stats
            else:
                print(f"⚠️  統計文件 {self.stats_file} 不存在")
                return None
        except Exception as e:
            print(f"❌ 加載統計信息失敗: {e}")
            return None
    
    def calculate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        計算數據的統計信息（均值、標準差、樣本數）
        
        Args:
            df (pd.DataFrame): 輸入數據
            
        Returns:
            Dict[str, Any]: 統計信息字典
        """
        stats = {}
        
        # 需要計算統計信息的列
        numeric_columns = ['temp', 'pressure', 'vibration']
        
        for col in numeric_columns:
            if col in df.columns:
                # 計算非空值的統計信息
                non_null_data = df[col].dropna()
                
                stats[col] = {
                    'mean': non_null_data.mean(),
                    'std': non_null_data.std(),
                    'count': len(non_null_data)
                }
                
                print(f"📊 {col}: 均值={stats[col]['mean']:.4f}, 標準差={stats[col]['std']:.4f}, 樣本數={stats[col]['count']}")
        
        return stats
    
    def _fill_missing_values(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        使用均值填充缺失值
        
        Args:
            df (pd.DataFrame): 輸入數據
            stats (Dict[str, Any]): 統計信息
            
        Returns:
            pd.DataFrame: 填充後的數據
        """
        df_filled = df.copy()
        
        for col in ['temp', 'pressure', 'vibration']:
            if col in df.columns and col in stats:
                mean_val = stats[col]['mean']
                # 使用均值填充缺失值
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        return df_filled
    
    def _z_score_normalize(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        使用Z-score標準化
        
        Args:
            df (pd.DataFrame): 輸入數據
            stats (Dict[str, Any]): 統計信息
            
        Returns:
            pd.DataFrame: 標準化後的數據
        """
        df_normalized = df.copy()
        
        for col in ['temp', 'pressure', 'vibration']:
            if col in df.columns and col in stats:
                mean_val = stats[col]['mean']
                std_val = stats[col]['std']
                
                # Z-score標準化: (x - mean) / std
                if std_val != 0:  # 避免除零錯誤
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                else:
                    print(f"⚠️  {col} 的標準差為0，跳過標準化")
        
        return df_normalized
    
    def process(self, csv_file: str, mode: str = 'training') -> pd.DataFrame:
        """
        處理CSV文件
        
        Args:
            csv_file (str): CSV文件路徑
            mode (str): 處理模式 ('training' 或 'eval')
            
        Returns:
            pd.DataFrame: 處理後的數據
        """
        try:
            # 讀取CSV文件
            df = pd.read_csv(csv_file)
            #print(f"✅ 成功讀取數據: {len(df)} 行")
            
            if mode == 'training':
                print("🔧 訓練模式: 計算統計信息並保存")
                
                # 計算統計信息
                self.stats = self.calculate_stats(df)
                
                # 保存統計信息
                self.save_stats(self.stats)
                
                # 填充缺失值
                df_filled = self._fill_missing_values(df, self.stats)
                
                # Z-score標準化
                df_normalized = self._z_score_normalize(df_filled, self.stats)
                
                print("✅ 訓練模式處理完成")
                return df_normalized
                
            elif mode == 'eval':
                #print("🔧 評估模式: 使用已保存的統計信息")
                
                # 加載統計信息
                self.stats = self.load_stats()
                
                if self.stats is None:
                    raise ValueError("無法加載統計信息，請先運行訓練模式")
                
                # 填充缺失值
                df_filled = self._fill_missing_values(df, self.stats)
                
                # Z-score標準化
                df_normalized = self._z_score_normalize(df_filled, self.stats)
                
                #print("✅ 評估模式處理完成")
                return df_normalized
                
            else:
                raise ValueError("模式必須是 'training' 或 'eval'")
                
        except Exception as e:
            print(f"❌ 數據處理失敗: {e}")
            return None

def main():
    """
    主函數，用於測試預處理器
    """
    # 創建預處理器實例
    preprocessor = DataPreprocessor()
    
    # 測試文件路徑（請根據實際情況修改）
    test_file = 'Data/training.csv'
    
    if os.path.exists(test_file):
        print("🧪 測試預處理器...")
        
        # 訓練模式
        print("\n📚 訓練模式測試:")
        df_training = preprocessor.process(test_file, mode='training')
        
        if df_training is not None:
            print(f"訓練模式結果: {df_training.shape}")
            print(df_training.head())
        
        # 評估模式
        print("\n🔍 評估模式測試:")
        df_eval = preprocessor.process(test_file, mode='eval')
        
        if df_eval is not None:
            print(f"評估模式結果: {df_eval.shape}")
            print(df_eval.head())
    else:
        print(f"⚠️  測試文件 {test_file} 不存在")

if __name__ == "__main__":
    main()