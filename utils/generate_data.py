import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

def generate_dummy_data(num_rows=300, 
                       normal_in_normal_range_prob=0.95,  # normal label時值在正常值域的機率
                       abnormal_in_normal_range_prob=0.3,  # abnormal label時值在正常值域的機率
                       null_prob=0.05):  # 產生空值的機率
    """
    Generate dummy sensor data with the following specifications:
    - timestamp: minute/five-minute intervals
    - temp: normal 45-50°C, abnormal >52 or <43
    - pressure: normal 1.00-1.05, abnormal >1.08 or <0.97
    - vibration: normal 0.02-0.04, abnormal >0.07
    - label: normal/abnormal based on sensor values
    
    Parameters:
    - normal_in_normal_range_prob: 當label為normal時，每個感測器值在正常值域的機率 (0-1)
    - abnormal_in_normal_range_prob: 當label為abnormal時，每個感測器值在正常值域的機率 (0-1)
    - null_prob: 產生空值的機率 (0-1)
    """
    
    # Start time
    start_time = datetime(2024, 6, 3, 19, 0, 0)
    
    data = []
    
    for i in range(num_rows):
        # Generate timestamp (every 1 minute)
        timestamp = start_time + timedelta(minutes=i)
        
        # Decide if this row should be normal or abnormal (80% normal, 20% abnormal)
        # 這邊可以調normal和abnormal的比例
        is_normal = random.random() < 0.8
        
        # 決定label
        if is_normal:
            label = "normal"
        else:
            label = "abnormal"
        
        # 決定是否產生空值
        temp_null = random.random() < null_prob
        pressure_null = random.random() < null_prob
        vibration_null = random.random() < null_prob
        
        # 根據label決定每個感測器是否在正常值域
        if is_normal:
            # normal label: 每個感測器都有normal_in_normal_range_prob的機率在正常值域
            temp_in_normal = random.random() < normal_in_normal_range_prob
            pressure_in_normal = random.random() < normal_in_normal_range_prob
            vibration_in_normal = random.random() < normal_in_normal_range_prob
        else:
            # abnormal label: 每個感測器都有abnormal_in_normal_range_prob的機率在正常值域
            temp_in_normal = random.random() < abnormal_in_normal_range_prob
            pressure_in_normal = random.random() < abnormal_in_normal_range_prob
            vibration_in_normal = random.random() < abnormal_in_normal_range_prob
        
        # 生成感測器數據
        if temp_null:
            temp = None
        else:
            if temp_in_normal:
                temp = random.uniform(45.0, 50.0)
            else:
                # 異常值域
                temp_choice = random.choice(['high', 'low'])
                if temp_choice == 'high':
                    temp = random.uniform(52.1, 55.0)
                else:
                    temp = random.uniform(40.0, 42.9)
        
        if pressure_null:
            pressure = None
        else:
            if pressure_in_normal:
                pressure = random.uniform(1.00, 1.05)
            else:
                # 異常值域
                pressure_choice = random.choice(['high', 'low'])
                if pressure_choice == 'high':
                    pressure = random.uniform(1.08, 1.15)
                else:
                    pressure = random.uniform(0.90, 0.97)
        
        if vibration_null:
            vibration = None
        else:
            if vibration_in_normal:
                vibration = random.uniform(0.02, 0.04)
            else:
                # 異常值域
                vibration = random.uniform(0.07, 0.12)
        
        # Round values for better readability (only if not null)
        if temp is not None:
            temp = round(temp, 1)
        if pressure is not None:
            pressure = round(pressure, 2)
        if vibration is not None:
            vibration = round(vibration, 2)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'temp': temp,
            'pressure': pressure,
            'vibration': vibration,
            'label': label
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='生成測試感測器數據')
    parser.add_argument('-o', '--output', default='testing.csv', 
                       help='輸出檔案名稱 (默認: testing.csv)')
    parser.add_argument('-n', '--num_rows', type=int, default=300,
                       help='生成數據行數 (默認: 300)')
    parser.add_argument('--normal_prob', type=float, default=0.8,
                       help='normal label時值在正常值域的機率 (默認: 0.8)')
    parser.add_argument('--abnormal_prob', type=float, default=0.3,
                       help='abnormal label時值在正常值域的機率 (默認: 0.3)')
    parser.add_argument('--null_prob', type=float, default=0.05,
                       help='產生空值的機率 (默認: 0.05)')

    args = parser.parse_args()
    
    # Generate data
    df = generate_dummy_data(
        num_rows=args.num_rows,
        normal_in_normal_range_prob=args.normal_prob,
        abnormal_in_normal_range_prob=args.abnormal_prob,
        null_prob=args.null_prob
    )
    
    # Save to CSV file
    df.to_csv(args.output, index=False)
    
    print("✅ 測試數據生成成功!")
    print(f"📁 輸出檔案: {args.output}")
    print(f"📊 總行數: {len(df)}")
    print(f"✅ 正常數據: {len(df[df['label'] == 'normal'])} 行")
    print(f"⚠️  異常數據: {len(df[df['label'] == 'abnormal'])} 行")
    print(f"📈 異常比例: {(len(df[df['label'] == 'abnormal']) / len(df) * 100):.1f}%")
    
    # 統計空值
    null_counts = df.isnull().sum()
    print(f"🔍 空值統計:")
    print(f"   - temp: {null_counts['temp']} 個空值")
    print(f"   - pressure: {null_counts['pressure']} 個空值")
    print(f"   - vibration: {null_counts['vibration']} 個空值")
    
    print(f"\n⚙️  使用參數:")
    print(f"   - normal label時每個感測器在正常值域機率: {args.normal_prob}")
    print(f"   - abnormal label時每個感測器在正常值域機率: {args.abnormal_prob}")
    print(f"   - 空值產生機率: {args.null_prob}")
    
    print("\n📋 前 10 行數據:")
    print(df.head(10))
    print("\n📋 後 10 行數據:")
    print(df.tail(10))