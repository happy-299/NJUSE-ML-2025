"""
Level 3 - AI Agent 运行脚本
使用 AI Agent 完成代码评审的四个任务

Usage:
    # 运行完整四任务流程
    python run_agent.py --tasks all --dataset quality --limit 10
    
    # 运行单个任务
    python run_agent.py --tasks task1 --dataset quality --limit 5
    
    # 使用本地模型
    python run_agent.py --llm-type local --dataset quality --limit 10 --full-pipeline
    
    # 使用 OpenAI API
    python run_agent.py --llm-type openai --model gpt-4o-mini --dataset quality
"""

import sys
from pathlib import Path

# 添加 level3 目录到 Python 路径
level3_dir = Path(__file__).parent
sys.path.insert(0, str(level3_dir))

# 导入并运行 main 模块
if __name__ == "__main__":
    from agent.main import main
    main()
