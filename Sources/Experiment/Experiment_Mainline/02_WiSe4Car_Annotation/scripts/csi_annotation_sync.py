"""
WiSe4Car 视频-CSI 时间同步与标注工具

功能：
1. 检查视频和 CSV 文件的时间同步
2. 可视化视频 + CSI 特征对齐效果
3. 从视频标注文件生成对齐的训练数据
4. 提供统计信息
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import cv2
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class VideoCSISyncChecker:
    """检查视频和 CSI 数据的时间同步"""
    
    def __init__(self, video_path: str, csv_directory: str):
        """
        初始化同步检查器
        
        Args:
            video_path: 视频文件路径 (AVI)
            csv_directory: CSV 文件所在目录
        """
        self.video_path = Path(video_path)
        self.csv_dir = Path(csv_directory)
        self.video_info = self._get_video_info()
        self.csv_info_list = self._get_csv_info()
        
    def _get_video_info(self) -> Dict:
        """从视频中提取时间信息"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps
        cap.release()
        
        return {
            "path": str(self.video_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "duration_mmss": self._seconds_to_mmss(duration_seconds)
        }
    
    def _get_csv_info(self) -> List[Dict]:
        """从 CSV 文件中提取时间跨度"""
        csv_files = sorted(self.csv_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"没有找到 CSV 文件: {self.csv_dir}")
        
        csv_info_list = []
        
        for csv_path in csv_files:
            try:
                # 只读取前后行来获取时间范围（加快速度）
                df_head = pd.read_csv(csv_path, sep=';', nrows=1)
                df_tail = pd.read_csv(csv_path, sep=';', skiprows=lambda x: x < 1 or x >= 2)
                
                df = pd.concat([df_head, df_tail])
                
                if 'timestamp' not in df.columns:
                    print(f"⚠️ {csv_path.name}: 没有找到 'timestamp' 列")
                    continue
                
                start_time = df['timestamp'].iloc[0]
                end_time = df['timestamp'].iloc[-1]
                total_records = len(pd.read_csv(csv_path, sep=';'))
                
                csv_info_list.append({
                    "filename": csv_path.name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "total_records": total_records
                })
                
            except Exception as e:
                print(f"❌ 读取 {csv_path.name} 时出错: {e}")
                continue
        
        return csv_info_list
    
    @staticmethod
    def _seconds_to_mmss(seconds: float) -> str:
        """将秒转换为 MM:SS 格式"""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"
    
    def check_sync_status(self) -> Dict:
        """检查同步状态"""
        print("\n" + "="*70)
        print("📹 视频-CSI 时间同步检查报告")
        print("="*70)
        
        print("\n【视频信息】")
        print(f"  文件: {self.video_path.name}")
        print(f"  FPS: {self.video_info['fps']:.2f}")
        print(f"  总帧数: {self.video_info['frame_count']}")
        print(f"  时长: {self.video_info['duration_mmss']} ({self.video_info['duration_seconds']:.2f}秒)")
        
        print(f"\n【CSV 文件信息】(共 {len(self.csv_info_list)} 个接收器)")
        
        for info in self.csv_info_list:
            print(f"\n  {info['filename']}")
            print(f"    时间范围: {info['start_time']} 到 {info['end_time']}")
            print(f"    总记录数: {info['total_records']}")
        
        # 检查都是否对齐
        if len(self.csv_info_list) > 1:
            start_times = [info['start_time'] for info in self.csv_info_list]
            print(f"\n【时间对齐检查】")
            
            if len(set(start_times)) == 1:
                print(f"  ✅ 所有 CSV 文件的开始时间一致")
            else:
                print(f"  ⚠️ CSV 文件开始时间不一致:")
                for info in self.csv_info_list:
                    print(f"    {info['filename']}: {info['start_time']}")
        
        print("\n" + "="*70)
        
        return {
            "video_info": self.video_info,
            "csv_info": self.csv_info_list,
            "status": "OK" if self.csv_info_list else "MISSING_CSV"
        }


class AnnotationToTrainingData:
    """将视频标注转换为 CSI 训练数据"""
    
    def __init__(self, csv_directory: str, annotation_json: str):
        """
        初始化转换器
        
        Args:
            csv_directory: CSV 文件目录
            annotation_json: 标注 JSON 文件路径
        """
        self.csv_dir = Path(csv_directory)
        self.annotation_data = self._load_annotation(annotation_json)
        self.csi_data = None
        
    def _load_annotation(self, json_path: str) -> Dict:
        """加载标注 JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_timestamp(self, ts_str: str) -> float:
        """
        将时间戳字符串转换为秒数（相对于开始时间）
        
        示例输入: "2024-06-18 12:19:06:586787"
        """
        try:
            # 分离毫秒部分
            parts = ts_str.split(':')
            if len(parts) == 7:  # HH:MM:SS:microseconds (实际上是纳秒/微秒)
                dt_str = ':'.join(parts[:3])
                microseconds = int(parts[3])
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp() + microseconds / 1e6
            else:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
        except Exception as e:
            print(f"⚠️ 时间戳解析失败: {ts_str}, 错误: {e}")
            return None
    
    def load_csi_data(self) -> pd.DataFrame:
        """加载并合并所有 CSV 文件"""
        csv_files = sorted(self.csv_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"没有找到 CSV 文件: {self.csv_dir}")
        
        print(f"📂 读取 {len(csv_files)} 个 CSI 文件...")
        
        # 读取第一个文件来获取基础时间戳
        df_list = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, sep=';')
                df['source_file'] = csv_path.name
                df_list.append(df)
            except Exception as e:
                print(f"❌ 读取 {csv_path.name} 失败: {e}")
        
        if not df_list:
            raise ValueError("无法读取任何 CSV 文件")
        
        self.csi_data = pd.concat(df_list, ignore_index=True)
        
        # 解析时间戳
        self.csi_data['timestamp_seconds'] = self.csi_data['timestamp'].apply(
            self._parse_timestamp
        )
        
        # 相对时间 (从第一条记录开始)
        start_time = self.csi_data['timestamp_seconds'].min()
        self.csi_data['relative_time'] = self.csi_data['timestamp_seconds'] - start_time
        
        print(f"  ✅ 加载了 {len(self.csi_data)} 条 CSI 记录")
        
        return self.csi_data
    
    def map_annotations_to_csi(self) -> pd.DataFrame:
        """将注释映射到 CSI 记录"""
        if self.csi_data is None:
            raise ValueError("请先调用 load_csi_data()")
        
        # 为每条 CSI 记录添加标注标签
        self.csi_data['action'] = 'unknown'
        self.csi_data['confidence'] = 0.0
        
        print("\n🏷️ 映射注释到 CSI 记录...")
        
        for annotation in self.annotation_data.get('annotations', []):
            start_second = annotation['start_time']
            end_second = annotation['end_time']
            action = annotation['action']
            confidence = annotation.get('confidence', 1.0)
            
            # 找到时间范围内的记录
            mask = (self.csi_data['relative_time'] >= start_second) & \
                   (self.csi_data['relative_time'] < end_second)
            
            self.csi_data.loc[mask, 'action'] = action
            self.csi_data.loc[mask, 'confidence'] = confidence
        
        labeled_count = len(self.csi_data[self.csi_data['action'] != 'unknown'])
        print(f"  ✅ 标注了 {labeled_count} 条记录 ({labeled_count/len(self.csi_data)*100:.1f}%)")
        
        return self.csi_data
    
    def get_statistics(self) -> Dict:
        """获取标注统计"""
        if self.csi_data is None:
            raise ValueError("请先调用 load_csi_data() 和 map_annotations_to_csi()")
        
        stats = {
            "total_records": len(self.csi_data),
            "labeled_records": len(self.csi_data[self.csi_data['action'] != 'unknown']),
            "action_distribution": self.csi_data[self.csi_data['action'] != 'unknown']['action'].value_counts().to_dict(),
            "time_span_seconds": self.csi_data['relative_time'].max()
        }
        
        print("\n📊 标注统计:")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  已标注: {stats['labeled_records']} ({stats['labeled_records']/stats['total_records']*100:.1f}%)")
        print(f"  时间跨度: {stats['time_span_seconds']:.2f} 秒")
        print(f"  动作分布:")
        for action, count in stats['action_distribution'].items():
            print(f"    - {action}: {count} 条 ({count/stats['labeled_records']*100:.1f}%)")
        
        return stats
    
    def export_training_data(self, output_csv: str):
        """导出为训练数据 CSV"""
        if self.csi_data is None:
            raise ValueError("请先调用 load_csi_data() 和 map_annotations_to_csi()")
        
        # 选择相关列
        export_cols = [
            'timestamp', 'relative_time', 'source_file', 'subcarrier', 
            'amplitude', 'phase', 'RSSI', 'action', 'confidence'
        ]
        
        export_data = self.csi_data[export_cols].copy()
        export_data.to_csv(output_csv, index=False)
        
        print(f"\n💾 导出训练数据: {output_csv}")
        print(f"  包含 {len(export_data)} 条记录")
        
        return export_data


def main():
    """主函数-示例用法"""
    
    # 示例 1: 检查同步
    print("【示例 1】检查视频-CSI 时间同步\n")
    
    video_path = "path/to/101_c1.avi"
    csv_dir = "path/to/101_c1/"
    
    try:
        sync_checker = VideoCSISyncChecker(video_path, csv_dir)
        sync_status = sync_checker.check_sync_status()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   请确保两个文件都存在")
    
    # 示例 2: 将标注转换为训练数据
    print("\n【示例 2】将标注转换为训练数据\n")
    
    annotation_json = "path/to/annotations_101_c1.json"
    output_csv = "path/to/training_data_101_c1.csv"
    
    try:
        converter = AnnotationToTrainingData(csv_dir, annotation_json)
        converter.load_csi_data()
        converter.map_annotations_to_csi()
        stats = converter.get_statistics()
        converter.export_training_data(output_csv)
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")


if __name__ == "__main__":
    main()
