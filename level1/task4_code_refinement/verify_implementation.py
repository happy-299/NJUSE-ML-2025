"""
Level 1 Task 4: 代码修复生成 - 验证脚本

用于验证任务四实现的完整性和正确性
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (NOT FOUND)")
        return False

def check_data_format():
    """检查数据格式"""
    print("\n" + "="*50)
    print("检查数据格式...")
    print("="*50)
    
    data_files = [
        "../../data/raw/ref-train.jsonl",
        "../../data/raw/ref-valid.jsonl", 
        "../../data/raw/ref-test.jsonl"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n检查 {data_file}:")
            try:
                # 读取前3行检查格式
                with open(data_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        
                        try:
                            data = json.loads(line.strip())
                            required_fields = ['old_hunk', 'comment', 'new']
                            
                            missing_fields = [field for field in required_fields 
                                            if field not in data]
                            
                            if missing_fields:
                                print(f"  行 {i+1}: 缺少字段 {missing_fields}")
                            else:
                                print(f"  行 {i+1}: ✓ 包含所需字段")
                                
                                # 显示字段长度
                                print(f"    old_hunk: {len(data['old_hunk'])} chars")
                                print(f"    comment: {len(data['comment'])} chars")
                                print(f"    new: {len(data['new'])} chars")
                                print(f"    language: {data.get('lang', 'unknown')}")
                        
                        except json.JSONDecodeError as e:
                            print(f"  行 {i+1}: JSON格式错误 - {e}")
                            
                print(f"  文件格式检查完成")
                        
            except Exception as e:
                print(f"  错误: {e}")
        else:
            print(f"✗ 数据文件不存在: {data_file}")

def check_imports():
    """检查必要的导入"""
    print("\n" + "="*50)
    print("检查Python包依赖...")
    print("="*50)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('json', 'JSON (内置)'),
        ('tqdm', 'TQDM')
    ]
    
    optional_packages = [
        ('nltk', 'NLTK (用于BLEU评估)'),
        ('rouge', 'ROUGE (用于ROUGE-L评估)'),
        ('codebleu', 'CodeBLEU (用于代码专用评估)'),
        ('bert_score', 'BERTScore (用于BERTScore评估)'),
        ('matplotlib', 'Matplotlib (用于可视化)')
    ]
    
    print("必需包:")
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} - 未安装")
    
    print("\n可选包:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"- {description} - 未安装 (可选)")

def check_project_structure():
    """检查项目结构"""
    print("\n" + "="*50)
    print("检查项目结构...")
    print("="*50)
    
    files_to_check = [
        ("README.md", "任务四说明文档"),
        ("QUICKSTART.md", "快速开始指南"),
        ("train.py", "训练脚本"),
        ("test.py", "测试脚本"),
        ("inference.py", "推理脚本"),
        ("preprocess_data.py", "数据预处理脚本"),
        ("sh/train.sh", "训练shell脚本"),
        ("sh/test.sh", "测试shell脚本"),
        ("sh/inference.sh", "推理shell脚本"),
        ("sh/interactive.sh", "交互推理shell脚本"),
        ("sh/quick_test.sh", "快速测试shell脚本"),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    # 检查其他重要文件
    print(f"\n检查项目配置文件:")
    check_file_exists("../../config.py", "项目配置文件")
    check_file_exists("../../utils/metrics.py", "评估指标文件")
    
    return all_exist

def test_data_loading():
    """测试数据加载功能"""
    print("\n" + "="*50)
    print("测试数据加载...")
    print("="*50)
    
    try:
        # 导入训练脚本中的数据加载函数
        from train import load_data
        from transformers import RobertaTokenizer
        
        # 加载tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # 尝试加载少量测试数据
        test_file = "../../data/raw/ref-test.jsonl"
        if os.path.exists(test_file):
            print(f"测试数据加载: {test_file}")
            dataset = load_data(test_file, tokenizer, 512, 128, max_samples=10)
            print(f"✓ 成功加载 {len(dataset)} 个样本")
            
            # 测试数据集的第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  样本数据键: {sample.keys()}")
                print(f"  source_ids 形状: {sample['source_ids'].shape}")
                print(f"  target_ids 形状: {sample['target_ids'].shape}")
        else:
            print(f"✗ 测试数据文件不存在: {test_file}")
            
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")

def test_metrics():
    """测试评估指标"""
    print("\n" + "="*50)
    print("测试评估指标...")
    print("="*50)
    
    try:
        sys.path.append("../../utils")
        from metrics import evaluate_task4, calculate_exact_match, calculate_bleu
        
        # 测试数据
        predictions = [
            "def add(a, b):\n    return a + b",
            "int x = 5;"
        ]
        references = [
            "def add(a, b):\n    return a + b",  # 完全匹配
            "int x = 10;"                        # 不匹配
        ]
        languages = ["python", "java"]
        
        # 测试任务四评估
        results = evaluate_task4(predictions, references, languages)
        print("✓ 任务四评估指标测试:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
            
        # 测试单独指标
        em = calculate_exact_match(predictions, references)
        bleu4 = calculate_bleu(predictions, references)
        print(f"\n✓ 单独指标测试:")
        print(f"  Exact Match: {em:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        
    except Exception as e:
        print(f"✗ 评估指标测试失败: {e}")

def test_shell_scripts():
    """测试shell脚本语法"""
    print("\n" + "="*50)
    print("检查Shell脚本语法...")
    print("="*50)
    
    shell_scripts = [
        "sh/train.sh",
        "sh/test.sh", 
        "sh/inference.sh",
        "sh/interactive.sh",
        "sh/quick_test.sh"
    ]
    
    for script in shell_scripts:
        if os.path.exists(script):
            try:
                # 在Windows上，我们只检查文件是否可读
                with open(script, 'r') as f:
                    content = f.read()
                    if content.strip():
                        print(f"✓ {script} - 文件内容正常")
                    else:
                        print(f"✗ {script} - 文件为空")
            except Exception as e:
                print(f"✗ {script} - 读取失败: {e}")
        else:
            print(f"✗ {script} - 文件不存在")

def create_sample_input():
    """创建示例输入文件"""
    print("\n" + "="*50)
    print("创建示例输入文件...")
    print("="*50)
    
    sample_data = [
        {
            "old_hunk": "def add(a, b):\n    return a + b",
            "comment": "Add type hints and input validation",
            "new": "def add(a: int, b: int) -> int:\n    if not isinstance(a, int) or not isinstance(b, int):\n        raise TypeError('Both arguments must be integers')\n    return a + b",
            "lang": "python"
        },
        {
            "old_hunk": "public int divide(int a, int b) {\n    return a / b;\n}",
            "comment": "Handle division by zero",
            "new": "public int divide(int a, int b) {\n    if (b == 0) {\n        throw new ArithmeticException(\"Division by zero\");\n    }\n    return a / b;\n}",
            "lang": "java"
        }
    ]
    
    output_file = "sample_input.jsonl"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in sample_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ 创建示例输入文件: {output_file}")
        print(f"  包含 {len(sample_data)} 个样本")
        
    except Exception as e:
        print(f"✗ 创建示例输入失败: {e}")

def main():
    """主函数"""
    print("Level 1 Task 4: 代码修复生成 - 实现验证")
    print("="*60)
    
    # 切换到任务四目录
    task4_dir = Path(__file__).parent
    os.chdir(task4_dir)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 执行各项检查
    structure_ok = check_project_structure()
    check_imports()
    check_data_format()
    test_data_loading()
    test_metrics()
    test_shell_scripts()
    create_sample_input()
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    if structure_ok:
        print("✓ 项目结构完整")
    else:
        print("✗ 项目结构存在问题")
    
    print("\n下一步操作:")
    print("1. 安装缺失的Python包:")
    print("   pip install torch transformers nltk rouge codebleu bert-score")
    print("\n2. 下载NLTK数据:")
    print("   python -c \"import nltk; nltk.download('punkt')\"")
    print("\n3. 开始训练:")
    print("   bash sh/train.sh")
    print("\n4. 测试模型:")
    print("   bash sh/quick_test.sh")
    print("\n5. 交互式推理:")
    print("   bash sh/interactive.sh")
    
    print(f"\n验证完成! 检查 {os.getcwd()} 目录下的sample_input.jsonl作为示例输入。")

if __name__ == "__main__":
    main()