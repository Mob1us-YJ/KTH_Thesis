"""
将标注结果导出为可分析的 CSV 格式
"""
import json
from pathlib import Path
import pandas as pd

def export_annotations_to_csv():
    """导出所有标注为 CSV 文件"""
    
    anno_dir = Path("Annotation_Workspace/Annotation_Result")
    anno_files = list(anno_dir.glob("*.json"))
    
    all_data = []
    
    # 采集所有标注数据
    for anno_file in sorted(anno_files):
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data['session_id']
        video_duration = data['video_duration_seconds']
        
        for anno in data['annotations']:
            all_data.append({
                'session_id': session_id,
                'action': anno['action'],
                'start_time': anno['start_time'],
                'end_time': anno['end_time'],
                'duration': anno['end_time'] - anno['start_time'],
                'video_duration': video_duration,
                'coverage_ratio': (anno['end_time'] - anno['start_time']) / video_duration * 100,
            })
    
    # 创建 DataFrame
    df = pd.DataFrame(all_data)
    
    # 保存为 CSV
    csv_file = Path("Annotation_Workspace/all_annotations.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"✅ CSV 文件已生成: {csv_file}")
    print(f"   • 总行数: {len(df)}")
    print(f"   • 列数: {len(df.columns)}\n")
    
    # 生成统计摘要
    print("📊 CSV 数据摘要:")
    print(df.head(10).to_string())
    print("\n...")
    print(df.tail(5).to_string())
    print(f"\n⚙️  数据统计:")
    print(df.describe().to_string())

if __name__ == "__main__":
    export_annotations_to_csv()
