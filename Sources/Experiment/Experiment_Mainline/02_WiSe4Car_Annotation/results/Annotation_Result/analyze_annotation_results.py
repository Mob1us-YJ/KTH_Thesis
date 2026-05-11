"""
分析所有 WiSe4Car 标注结果
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime

def analyze_annotations():
    """分析所有标注结果"""
    
    anno_dir = Path("Annotation_Workspace/Annotation_Result")
    anno_files = list(anno_dir.glob("*.json"))
    
    print("\n" + "="*90)
    print("📊 WiSe4Car 视频标注结果分析总结")
    print("="*90 + "\n")
    
    # 统计数据
    total_sessions = len(anno_files)
    total_annotations = 0
    action_counts = defaultdict(int)
    session_data = []
    all_annotations = []
    
    # 读取所有标注文件
    for anno_file in sorted(anno_files):
        with open(anno_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data['session_id']
        video_duration = data['video_duration_seconds']
        anno_count = data['annotation_count']
        
        total_annotations += anno_count
        
        # 统计每个会话的信息
        session_info = {
            'session_id': session_id,
            'duration': video_duration,
            'num_annotations': anno_count,
            'annotated_time': 0,
            'actions': []
        }
        
        for anno in data['annotations']:
            action = anno['action']
            action_counts[action] += 1
            duration = anno['end_time'] - anno['start_time']
            session_info['annotated_time'] += duration
            session_info['actions'].append({
                'action': action,
                'start_time': anno['start_time'],
                'end_time': anno['end_time'],
                'duration': duration
            })
            
            all_annotations.append({
                'session_id': session_id,
                'action': action,
                'start_time': anno['start_time'],
                'end_time': anno['end_time'],
                'duration': duration
            })
        
        session_data.append(session_info)
    
    # 打印统计信息
    print(f"📋 基本统计:")
    print(f"   • 总会话数: {total_sessions}")
    print(f"   • 总标注数: {total_annotations}")
    print(f"   • 平均每会话标注数: {total_annotations/total_sessions:.2f}")
    print()
    
    # 动作分布
    print(f"🎯 动作类型分布:")
    print(f"   {'动作类型':<20} {'数量':<10} {'占比':<10}")
    print("   " + "-"*40)
    
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    for action, count in sorted_actions:
        percentage = count / total_annotations * 100
        print(f"   {action:<20} {count:<10} {percentage:>6.1f}%")
    
    print()
    
    # 按会话类型分组统计（提取会话前缀）
    print(f"📂 按会话类型分布:")
    session_types = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'total_annotations': 0})
    
    for session in session_data:
        session_id = session['session_id']
        prefix = session_id.split('_')[0]  # 提取数字前缀
        session_types[prefix]['count'] += 1
        session_types[prefix]['total_duration'] += session['duration']
        session_types[prefix]['total_annotations'] += session['num_annotations']
    
    print(f"   {'类型':<10} {'会话数':<10} {'总时长(s)':<15} {'标注数':<10}")
    print("   " + "-"*45)
    
    for stype in sorted(session_types.keys()):
        info = session_types[stype]
        print(f"   {stype:<10} {info['count']:<10} {info['total_duration']:<15.1f} {info['total_annotations']:<10}")
    
    print()
    
    # 标注时间统计
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        
        print(f"⏱️  标注时间统计:")
        print(f"   • 总标注时长: {df['duration'].sum():.2f} 秒")
        print(f"   • 平均标注时长: {df['duration'].mean():.2f} 秒")
        print(f"   • 最长标注: {df['duration'].max():.2f} 秒")
        print(f"   • 最短标注: {df['duration'].min():.2f} 秒")
        print(f"   • 中位数: {df['duration'].median():.2f} 秒")
        print()
        
        print(f"🎬 按动作类型的时间统计:")
        print(f"   {'动作类型':<20} {'平均时长(s)':<15} {'总时长(s)':<15}")
        print("   " + "-"*50)
        
        for action in sorted(action_counts.keys()):
            action_df = df[df['action'] == action]
            avg_duration = action_df['duration'].mean()
            total_duration = action_df['duration'].sum()
            print(f"   {action:<20} {avg_duration:<15.2f} {total_duration:<15.2f}")
    
    print()
    
    # 会话详细统计
    print(f"📍 活跃度最高的会话（Top 10）:")
    print(f"   {'会话ID':<15} {'标注数':<10} {'总时长(s)':<15} {'覆盖率(%)':<10}")
    print("   " + "-"*50)
    
    session_data_sorted = sorted(session_data, key=lambda x: x['num_annotations'], reverse=True)
    for i, session in enumerate(session_data_sorted[:10]):
        coverage = (session['annotated_time'] / session['duration'] * 100) if session['duration'] > 0 else 0
        print(f"   {session['session_id']:<15} {session['num_annotations']:<10} {session['annotated_time']:<15.2f} {coverage:<10.1f}%")
    
    print()
    
    # 保存详细报告
    report_file = Path("Annotation_Workspace/Annotation_Report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("WiSe4Car 视频标注结果分析报告\n")
        f.write("="*90 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"分析会话数: {total_sessions}\n")
        f.write(f"总标注数: {total_annotations}\n\n")
        
        f.write("动作类型分布:\n")
        for action, count in sorted_actions:
            percentage = count / total_annotations * 100
            f.write(f"  {action}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n会话类型分布:\n")
        for stype in sorted(session_types.keys()):
            info = session_types[stype]
            f.write(f"  {stype}: {info['count']} 会话, 总标注 {info['total_annotations']} 个\n")
    
    print(f"✅ 详细报告已保存到: Annotation_Workspace/Annotation_Report.txt\n")
    
    print("="*90)
    print("✨ 分析完成！")
    print("="*90 + "\n")

if __name__ == "__main__":
    analyze_annotations()
