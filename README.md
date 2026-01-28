# APIGen-MT-5k Task Generation & Testing System

<div align="center">

**A comprehensive framework for generating and testing multi-turn customer service tasks**

[中文文檔](#中文文檔) | [English Documentation](#english-documentation)

</div>

---

## 📋 目錄 / Table of Contents

- [中文文檔](#中文文檔)
  - [專案概述](#專案概述)
  - [目錄結構](#目錄結構)
  - [快速開始](#快速開始)
  - [核心組件](#核心組件)
  - [使用示例](#使用示例)
  - [配置系統](#配置系統)
  - [模型訓練與測試](#模型訓練與測試)
  - [故障排除](#故障排除)
- [English Documentation](#english-documentation)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Quick Start](#quick-start)
  - [Core Components](#core-components)
  - [Usage Examples](#usage-examples)
  - [Configuration System](#configuration-system)
  - [Model Training & Testing](#model-training--testing)
  - [Troubleshooting](#troubleshooting)

---

# 中文文檔

## 專案概述

**APIGen-MT-5k** 是一個先進的多輪任務生成與測試系統，專為客戶服務場景設計。該系統實現了 **AgentFlow** 多輪迭代架構，支援雙模型協作（GPT-4o + GPT-OSS-120B），提供完整的任務生成、驗證、測試和分析流程。

### 🎯 核心特性

- **🔄 AgentFlow 架構**: 多輪迭代生成，通過規劃器、執行器、驗證器和生成器協作，生成高質量任務
- **🤖 雙模型系統**: GPT-4o（用戶模型）+ GPT-OSS-120B（助手模型）協作，提升任務質量
- **✅ 智能驗證**: 自動驗證任務的數據一致性、用戶ID一致性和工具調用正確性
- **📊 全面測試**: 支援並行測試、多種評估指標和結果可視化
- **🔧 靈活配置**: 通過 `configs.py` 集中管理所有配置參數
- **📈 詳細報告**: 生成 JSON、CSV 和可視化報告，支援深度分析
- **🎓 模型訓練**: 支援 GPT-OSS 20B 模型的強化學習訓練（GRPO + SFT），可自定義訓練客服任務模型

### 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    APIGen-MT-5k System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │ Task Pipeline│────────────────→│Task Generator│        │
│  │  (Pipeline)  │                  │ (AgentFlow)  │        │
│  └──────────────┘                  └──────┬───────┘        │
│         │                                 │                │
│         │                                 ↓                │
│         │                          ┌──────────────┐        │
│         │                          │  Validator   │        │
│         │                          │ (Consistency │        │
│         │                          │   Checker)   │        │
│         │                          └──────┬───────┘        │
│         │                                 │                │
│         ↓                                 ↓                │
│  ┌──────────────┐                  ┌──────────────┐        │
│  │ Task Tester  │←─────────────────│Generated Task│        │
│  │ (Dual Model) │                  │    (JSON)    │        │
│  └──────┬───────┘                  └──────────────┘        │
│         │                                                  │
│         ↓                                                  │
│  ┌──────────────┐                                          │
│  │   Results    │                                          │
│  │ (JSON/CSV/   │                                          │
│  │Visualization)│                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
APIGen-MT-5k/
├── task_tester.py              # 任務測試器（支援雙模型）
├── task_generator.py           # 任務生成器（支援 AgentFlow）
├── task_pipeline.py            # 任務生成管道
├── configs.py                  # 配置管理
├── data_reader.py              # 數據讀取工具
│
├── envs/                       # 環境數據
│   └── retail/                 # 零售環境
│       ├── users.json          # 用戶數據
│       ├── orders.json         # 訂單數據
│       ├── products.json       # 產品數據
│       └── tools/              # 工具實現
│
├── generated_tasks/            # 生成的任務
│   ├── Sampled_Tasks.json
│   ├── test_agentflow.json
│   └── test_direct.json
│
├── results/                    # 測試結果
│   ├── test_results_*.json
│   ├── test_summary_*.json
│   ├── test_summary_*.csv
│   └── test_visualization_*.png
│
├── scripts/                    # 輔助腳本
│   ├── analysis_helpers.py              # 分析輔助函數庫
│   ├── analyze_generated_tasks.py       # 分析生成任務的統計信息
│   ├── analyze_real_data_references.py  # 分析真實數據引用情況
│   ├── analyze_successful_tasks.py      # 分析成功任務的模式
│   ├── analyze_unsuccessful_tasks.py     # 分析失敗任務的模式
│   ├── compare_blueprint_agentflow.py   # 比較兩種生成方法
│   ├── filter_successful_sample_tasks.py # 過濾成功和失敗的任務
│   └── find_specific_failed_tasks.py    # 查找特定失敗任務
│
├── utils/                      # 工具函數
│   ├── arg_normalizer.py       # 參數標準化工具
│   └── __init__.py
│
├── gpt_oss_20b_tau_bench_rl_training.ipynb  # 模型訓練 notebook
├── dataset.jsonl               # 訓練數據集（800任務）
├── 3.5k.jsonl                  # 擴展數據集（3500+任務）
├── server.py                   # 模型部署服務器（FastAPI）
├── run_training.sh             # 訓練腳本
├── dgx_train.slurm             # SLURM 訓練配置
├── training_configs.ini        # 訓練配置文件
│
├── README.md                   # 本文檔
├── 模型訓練指南.md              # 模型訓練完整指南
├── 模型測試指南.md              # 模型測試部署指南
├── AGENTFLOW_README.md        # AgentFlow 詳細文檔
├── Compare_agentflow_direct.md # 方法對比分析
└──  SUMMARY.md                  # 專案摘要
```

## 快速開始

### 1. 安裝依賴

```bash
# 必需依賴
pip install openai

# 可選依賴（用於進度條和可視化）
pip install tqdm pandas matplotlib
```

### 2. 配置 API 密鑰

編輯 `configs.py` 文件，設置你的 API 密鑰：

```python
@dataclass
class TauBenchConfig:
    # GPT-4o (用戶模型)
    user_model: str = "gpt-4o"
    user_api_key: str = "your-gpt4o-api-key"
    user_base_url: str = "https://api.openai.com/v1"
    
    # GPT-OSS-120B (助手模型)
    default_model: str = "vllm-a40-gpt-oss-120b"
    default_api_key: str = "your-gpt-oss-api-key"
    default_base_url: str = "https://your-api-endpoint/api"
```

### 3. 基本使用

```bash
# 生成任務（AgentFlow 模式）
python task_pipeline.py --num-tasks 10 --agentflow --output generated_tasks/my_tasks.json

# 測試任務（雙模型）
python task_tester.py --tasks generated_tasks/my_tasks.json --dual-model --verbose --save-results --visualize

# 查看結果
ls results/
```

## 核心组件

### 1. Task Tester (`task_tester.py`)

任務測試器，支持雙模型測試器，支持雙模型協作和多種測試模式。

#### 主要功能

- ✅ **雙模型測試**: GPT-4o 增強查詢 + GPT-OSS-120B 執行任務
- ✅ **並行測試**: 多線程並行處理，提升測試效率
- ✅ **多種模式**: 真實執行、模擬執行、僅驗證
- ✅ **詳細指標**: 精確度、召回率、F1分數、輸出匹配率等
- ✅ **結果可視化**: 自動生成圖表和統計報告

#### 命令行參數

**輸入選項**:
```bash
--tasks FILE                    # 要測試的任務 JSON 文件
```

**雙模型選項**:
```bash
--dual-model                    # 啟用雙模型方法
--enhance-query                 # 使用用户模型增强查询
```

**助手模型配置** (GPT-OSS-120B):
```bash
--model MODEL                   # 模型名稱
--api-key KEY                   # API 密鑰
--base-url URL                  # API 基礎URL
```

**用户模型配置** (GPT-4o):
```bash
--user-model MODEL              # 用户模型名稱
--user-api-key KEY              # 用户模型 API 密鑰
--user-base-url URL             # 用户模型 API 基礎URL
```

**工具配置**:
```bash
--envs-path PATH                # 零售環境路徑（默認: envs/retail）
```

**執行選項**:
```bash
--threads N                     # 並行線程數（默認: 1）
--verbose                       # 詳細輸出
--dry-run                       # 模擬運行（不調用 LLM）
--validate-only                 # 僅驗證任務（不調用模型）
```

**輸出選項**:
```bash
--output-dir DIR                # 結果保存目錄（默認: results）
--save-results                  # 保存詳細結果
--visualize                     # 生成可視化圖表
```

#### 使用示例

```bash
# 基礎測試
python task_tester.py --tasks generated_tasks/my_tasks.json --verbose

# 雙模型測試 + 可視化
python task_tester.py \
    --tasks generated_tasks/my_tasks.json \
    --dual-model \
    --enhance-query \
    --threads 4 \
    --save-results \
    --visualize

# 僅驗證（不調用模型）
python task_tester.py --tasks generated_tasks/my_tasks.json --validate-only

# 模擬運行（快速測試）
python task_tester.py --tasks generated_tasks/my_tasks.json --dry-run --verbose
```

### 2. Task Generator (`task_generator.py`)

任務生成器，支援 AgentFlow 多輪迭代生成和直接生成兩種模式。

#### AgentFlow 架構

AgentFlow 是一個多輪迭代生成架構，包含以下組件：

```
Turn t: Query (q) + Knowledge (K) + Memory (M^t)
    ↓
  Planner (π_θ) → Actions (a^t)
    ↓
  Executor → Commands (c^t) + Results (e^t)
    ↓
  Verifier → Analysis + Status (v^t)
    ↓
  Generator (if complete) → Answer (o)
    ↓
  Memory^(t+1) (accumulated context)
```

**組件說明**:

1. **Planner（規劃器）**: 分析查詢和上下文，制定行動計劃
2. **Executor（執行器）**: 執行計劃的動作，獲取結果
3. **Verifier（驗證器）**: 驗證執行結果，決定下一步行動
4. **Generator（生成器）**: 在任務完成時生成最終輸出
5. **Memory（內存）**: 跨輪次累積上下文信息

#### 使用示例

```python
from task_generator import TauBenchOpenAIGenerator

# 創建生成器
generator = TauBenchOpenAIGenerator("envs/retail")

# 使用 AgentFlow 生成單個任務
result = generator.generate_task_with_agentflow(
    max_turns=5,
    include_metadata=True
)

# 批量生成任務
tasks = generator.generate_diverse_tasks(
    num_tasks=10,
    use_agentflow=True
)

# 保存任务
generator.save_tasks_to_file(tasks, "generated_tasks/my_tasks.json")
```

詳細的 AgentFlow 文檔请参考 [AGENTFLOW_README.md](AGENTFLOW_README.md)

### 3. Task Pipeline (`task_pipeline.py`)

任務生成管道，集成了生成、驗證、審查和优=優化的完整流程。

#### 主要功能

- 🔄 **迭代優化**: 自動進行多次迭代，直到任務通過驗證
- 👥 **審查委員會**: 多個審查器評估任務質量
- 🔍 **用戶ID驗證**: 確保所有動作使用一致的用戶ID
- 📊 **統計報告**: 詳細的生成統計和失敗原因分析
- 🎯 **場景多樣性**: 確保生成任務覆蓋多種場景類型

#### 命令行參數

```bash
--num-tasks N                   # 生成任務數量（默認: 3）
--max-iterations N              # 每个任務最大迭代次數（默認: 3）
--output FILE                   # 輸出文件路徑（默認: generated_tasks/Sampled_Tasks.json）
--no-user-id-validation         # 禁用用户ID一致性驗證
--committee-size N              # 審查委員會大小（默認: 3）
--agentflow                     # 使用 AgentFlow 多輪生成
--agentflow-turns N             # AgentFlow 最大輪數（默認: 5）
```

#### 使用示例

```bash
# 基礎生成（直接模式）
python task_pipeline.py --num-tasks 10 --output generated_tasks/my_tasks.json

# 使用 AgentFlow 生成
python task_pipeline.py --num-tasks 10 --agentflow --agentflow-turns 5 --output generated_tasks/agentflow_tasks.json

# 增加迭代次數和委員會大小
python task_pipeline.py --num-tasks 5 --max-iterations 5 --committee-size 5 --output generated_tasks/high_quality_tasks.json

# 禁用用戶ID驗證（僅在特殊情況下使用）
python task_pipeline.py --num-tasks 10 --no-user-id-validation --output generated_tasks/tasks.json
```

#### 管道流程

```
1. 初始化
   ↓
2. 生成任務（AgentFlow/Direct）
   ↓
3. 數據驗證
   ├─ 用戶存在性
   ├─ 訂單存在性
   ├─ 用戶-訂單匹配
   └─ 工具參數有效性
   ↓
4. 審查委員會評審
   ├─ 審查器 1
   ├─ 審查器 2
   └─ 審查器 3
   ↓
5. 決策
   ├─ 通過 → 保存任務
   ├─ 未通過 → 生成反饋
   └─ 重新生成（最多 max_iterations 次）
   ↓
6. 統計報告
```

### 4. Configuration System (`configs.py`)

集中式配置管理系統。

#### 主要配置項

```python
@dataclass
class TauBenchConfig:
    # 環境路徑
    envs_path: str = "envs/retail"
    
    # 模型配置
    user_model: str = "gpt-4o"                    # 用戶模型
    user_api_key: str = "your-key"               # 用戶模型密鑰
    user_base_url: str = "https://api.openai.com/v1"
    
    default_model: str = "vllm-a40-gpt-oss-120b" # 助手模型
    default_api_key: str = "your-key"            # 助手模型密鑰
    default_base_url: str = "https://your-endpoint/api"
    
    # 生成配置
    num_tasks: int = 10                          # 默認任務數量
    temperature: float = 0.0                     # 溫度參數
    max_tokens: int = 16384                      # 最大令牌數
    max_retries: int = 3                         # 最大重試次數
    timeout: int = 180                           # 超時時間（秒）
    
    # 驗證配置
    test_timeout: int = 180                      # 測試超時
    max_workers: int = 2                         # 最大工作線程
    
    # 輸出配置
    output_dir: str = "tau_bench_results"        # 輸出目錄
    save_visualizations: bool = True             # 保存可視化
    
    # 質量指標閾值
    min_success_rate: float = 0.7                # 最低成功率
    min_action_recall: float = 0.6               # 最低動作召回率
    min_action_precision: float = 0.6            # 最低動作精確率
    
    # 場景配置
    scenario_keys: List[str] = [                 # 場景類型
        'order_cancellation',
        'order_modification',
        'item_return',
        'item_exchange',
        'address_change',
        'payment_update',
        'order_inquiry',
        'product_inquiry',
    ]
```

## 使用示例

### 完整工作流

```bash
# 步驟 1: 使用 AgentFlow 生成高質量任务
python task_pipeline.py \
    --num-tasks 20 \
    --agentflow \
    --agentflow-turns 5 \
    --max-iterations 3 \
    --committee-size 3 \
    --output generated_tasks/production_tasks.json

# 步驟 2: 使用雙模型測試任務
python task_tester.py \
    --tasks generated_tasks/production_tasks.json \
    --dual-model \
    --enhance-query \
    --threads 8 \
    --save-results \
    --visualize \
    --output-dir results/production_test

# 步骤 3: 查看结果
ls results/production_test/
# test_results_*.json      - 詳細測試結果
# test_summary_*.json      - 彙總報告
# test_summary_*.csv       - CSV 格式
# test_visualization_*.png - 可視化圖表
```

### 對比 AgentFlow vs Direct 模式

```bash
# 使用腳本對比兩種方法
python scripts/compare_blueprint_agentflow.py \
    --num-tasks 10 \
    --direct-out generated_tasks/test_direct.json \
    --agent-out generated_tasks/test_agentflow.json \
    --agentflow-turns 5
```

詳細的對比分析請參考 [Compare_agentflow_direct.md](Compare_agentflow_direct.md)

### 分析数据引用

```bash
# 分析任務中的真實數據引用
python scripts/analyze_real_data_references.py \
    generated_tasks/my_tasks.json \
    --envs-path envs/retail \
    --output analysis_report.json
```

## 配置系統

### 修改配置

你可以通過以下方式修改配置：

1. **直接編輯 `configs.py`**（推薦用於永久更改）
2. **命令行參數覆蓋**（用於臨時更改）
3. **環境變量**（用於敏感信息）

### 配置優先級
```
命令行參數 > configs.py 設置 > 默認值
```

### 示例：自定義配置

```python
# 在代碼中使用自定義配置
from configs import TauBenchConfig

config = TauBenchConfig()
config.num_tasks = 50
config.temperature = 0.7
config.max_workers = 4

# 使用自定義配置
from task_pipeline import TaskConfigurationPipeline, PipelineConfig

pipeline_config = PipelineConfig(
    envs_path=config.envs_path,
    max_iterations=5,
    committee_size=5,
    use_agentflow=True
)

pipeline = TaskConfigurationPipeline(pipeline_config)
```

## 輸出說明

### 1. 任務文件 (JSON)

```json
[
  {
    "success": true,
    "task": {
      "q": "我需要取消訂單 #W123456",
      "agt": [
        {
          "name": "find_user_id_by_email",
          "arguments": {"email": "user@example.com"}
        },
        {
          "name": "get_order_details",
          "arguments": {"user_id": "U123", "order_id": "#W123456"}
        },
        {
          "name": "cancel_pending_order",
          "arguments": {"user_id": "U123", "order_id": "#W123456"}
        }
      ],
      "ogt": ["您的訂單 #W123456 已成功取消"]
    },
    "metadata": {
      "generation_method": "agentflow",
      "turns": 3,
      "confidence": 0.95
    },
    "validation_report": {
      "valid": true,
      "missing": [],
      "suggestions": []
    }
  }
]
```

### 2. 測試結果 (JSON)

```json
{
  "summary": {
    "total_tasks": 20,
    "successful": 18,
    "success_rate": 0.90,
    "avg_action_precision": 0.92,
    "avg_action_recall": 0.88,
    "avg_action_f1": 0.90,
    "exact_action_matches": 15
  },
  "results": [
    {
      "task_id": "task_001",
      "success": true,
      "action_precision": 1.0,
      "action_recall": 0.85,
      "action_f1": 0.92,
      "execution_time": 3.45,
      "model_response_time": 2.10,
      "tool_execution_time": 1.35
    }
  ]
}
```

### 3. CSV 彙總

| task_id | success | action_precision | action_recall | action_f1 | execution_time |
|---------|---------|------------------|---------------|-----------|----------------|
| task_001| True    | 1.00             | 0.85          | 0.92      | 3.45           |
| task_002| True    | 0.90             | 0.95          | 0.92      | 4.12           |

### 4. 可視化圖表

- **成功率分析**: 餅圖顯示成功/失敗任務比例
- **性能指標**: 柱狀圖顯示精確率、召回率、F1分數
- **時間分佈**: 直方圖顯示執行時間分佈
- **動作匹配**: 熱力圖顯示動作匹配情況

## 模型訓練與測試

本專案支援 **GPT-OSS 20B** 模型的強化學習訓練，可自定義訓練客服任務模型。訓練採用兩階段流程：

### 📚 完整文檔

- **[模型訓練指南.md](模型訓練指南.md)** - 詳細的模型訓練教程
- **[模型測試指南.md](模型測試指南.md)** - 模型部署和測試指南

### 🛠️ 訓練環境設置

訓練環境支持多種配置，選擇適合您的設置：

#### 本地環境（推薦用於開發）

```bash
# 標準安裝
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 針對 A100/H100 GPU（更好性能）
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

#### NVIDIA DGX Spark（大規模訓練）

用於訓練 200B 參數以下的大型模型（如 gpt-oss-120b）：

```bash
# 1. 下載並構建 Docker 鏡像
wget -O Dockerfile "https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark"
docker build -f Dockerfile -t unsloth-dgx-spark .

# 2. 啟動容器
docker run -it --gpus=all --net=host --ipc=host \
    -v $(pwd):$(pwd) -w $(pwd) unsloth-dgx-spark

# 3. 在容器內訓練
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**內存需求**：
- GPT-OSS 20B: 24-40 GB VRAM（4-bit 量化）
- GPT-OSS 120B: ~68 GB 統一內存（QLoRA 4-bit）

**資源**：
- 📚 [Unsloth DGX Spark 文檔](https://unsloth.ai/docs/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)
- 🐳 [DGX Spark Dockerfile](https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark)

#### 雲平台

**Google Colab**（免費 T4 GPU）：
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**AWS/Azure/GCP**（A100/H100）：
```bash
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

**詳細設置說明請參考** [模型訓練指南.md](模型訓練指南.md)

### 🎓 訓練流程概覽

#### 第一階段：SFT 預訓練（30-60 分鐘）

使用監督式微調讓模型學習 JSON 格式和基本工具選擇：

```bash
# 在專案根目錄下
jupyter notebook gpt_oss_20b_tau_bench_rl_training.ipynb

# 執行 Cell 1-16（SFT 訓練）
```

**特點**：
- 使用 800 個標註樣本
- 學習正確的工具調用格式
- JSON 有效率達 ~95%
- 動作準確率達 ~90%

#### 第二階段：GRPO 訓練（2-4 小時）

使用強化學習優化策略：

```bash
# 繼續執行 Cell 17-21（GRPO 訓練）
```

**獎勵機制**：
- ✅ +1.0 每個正確的動作名稱
- ✅ +0.5 每個正確的參數集
- ✅ +0.5 正確的順序
- ❌ -0.3 每個錯誤動作
- ❌ -0.4 每個缺失的必要參數

**預期結果**：
- 完全匹配率：70-85%
- 動作 F1 分數：75-90%
- 獎勵隨訓練增加

### 🚀 模型部署與測試

#### 1. 部署模型服務

訓練完成後，模型保存在專案目錄中（默認為 `outputs/` 或自定義路徑）

**選項 A：使用 FastAPI（推薦）**

```bash
# 啟動模型服務
python server.py

# 服務運行在 http://localhost:8000
```

**選項 B：使用 vLLM（高性能）**

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model outputs/merged_model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16
```

**選項 C：使用 Ollama（簡單部署）**

```bash
# 創建 Modelfile
echo 'FROM outputs/merged_model' > Modelfile.gpt-oss

# 創建並運行模型
ollama create gpt-oss-20b-tau -f Modelfile.gpt-oss
ollama serve
```

#### 2. 配置測試環境

編輯 `configs.py`，指向本地模型服務：

```python
@dataclass
class TauBenchConfig:
    # 使用訓練好的本地模型
    default_model: str = "gpt-oss-20b-tau"
    default_api_key: str = "EMPTY"  # 本地服務不需要密鑰
    default_base_url: str = "http://localhost:8000/v1"
```

#### 3. 運行測試

```bash
# 測試訓練好的模型
python task_tester.py \
    --tasks generated_tasks/my_tasks.json \
    --model gpt-oss-20b-tau \
    --base-url http://localhost:8000/v1 \
    --verbose \
    --save-results \
    --visualize

# 查看結果
ls results/
```

### 📊 性能評估

訓練完成後，可使用以下指標評估模型：

```python
import json

# 加載測試結果
with open('results/test_results_*.json', 'r') as f:
    results = json.load(f)

# 計算成功率
total = len(results['results'])
success = sum(1 for r in results['results'] if r['reward'] >= 0.99)
print(f"成功率: {success/total:.2%}")
print(f"平均動作 F1: {results['summary']['avg_action_f1']:.2f}")
print(f"平均動作精確率: {results['summary']['avg_action_precision']:.2f}")
print(f"平均動作召回率: {results['summary']['avg_action_recall']:.2f}")
```

### 🔧 訓練文件結構

```
APIGen-MT-5k/
├── gpt_oss_20b_tau_bench_rl_training.ipynb  # 主訓練 notebook
├── dataset.jsonl                             # 訓練數據（800 任務）
├── 3.5k.jsonl                                # 擴展數據集（3500+ 任務）
├── server.py                                 # FastAPI 服務器
├── run_training.sh                           # 訓練腳本
├── dgx_train.slurm                           # SLURM 配置
├── training_configs.ini                      # 訓練配置
├── 模型訓練指南.md                            # 訓練指南
└── 模型測試指南.md                            # 測試指南

# 訓練後會生成（根據配置）：
outputs/
├── checkpoint-*/                             # 訓練檢查點
├── final_model/                              # 最終模型（LoRA）
└── merged_model/                             # 合併後的完整模型
```

### 💡 訓練建議

**GPU 需求**：
- 建議使用 A100 或 H100（40GB+ VRAM）
- 最低要求：A40 或 V100（24GB VRAM）
- 訓練時間：4-6 小時（取決於 GPU）

**優化技巧**：
1. **增加數據量**：使用 `3.5k.jsonl`（3500+ 任務）獲得更好效果
2. **調整 LoRA rank**：增加到 32 或 64 提升模型容量
3. **調整學習率**：嘗試 1e-5, 5e-6, 2e-5
4. **批量大小**：根據 VRAM 調整 `per_device_train_batch_size`

**常見問題**：
- **CUDA 記憶體不足**：減少批量大小或序列長度
- **訓練不收斂**：調整學習率或增加 warm-up steps
- **準確率低（<60%）**：增加訓練數據或調整獎勵函數
- **訓練太慢**：使用多 GPU 或減少 epoch

更多詳細信息請參考：
- 📖 [模型訓練指南.md](模型訓練指南.md) - 完整訓練教程
- 📖 [模型測試指南.md](模型測試指南.md) - 部署和測試指南
- 📊 [SUMMARY.md](SUMMARY.md) - 專案摘要和概覽

## 故障排除

### 常見問題

#### 1. API 連接失敗

**問題**: `Connection refused` 或 `API key invalid`

**解決方案**:
- 檢查 `configs.py` 中的 API 密鑰和 URL
- 驗證網絡連接
- 確認 API 服務正常運行

```bash
# 測試 API 連接
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.openai.com/v1/models
```

#### 2. 工具執行失敗

**問題**: `Tool 'xxx' not found` 或 `Tool execution failed`

**解決方案**:
- 確認 `envs/retail/` 目錄存在
- 檢查 `tools/` 目錄中的工具實現
- 驗證數據文件（users.json, orders.json, products.json）格式正確

```bash
# 驗證數據文件
python -c "import json; json.load(open('envs/retail/users.json'))"
```

#### 3. 任務驗證失敗

**問題**: `Validation failed` 或 `User not found`

**解決方案**:
- 檢查任務中的用戶ID和訂單ID是否存在於數據文件中
- 使用 `--validate-only` 選項單獨運行驗證
- 查看詳細的驗證報告

```bash
# 僅驗證任務
python task_tester.py --tasks my_tasks.json --validate-only --verbose
```

#### 4. 内存不足

**問題**: `MemoryError` 或系統變慢

**解決方案**:
- 減少 `--threads` 參數值
- 分批處理任務
- 增加系統內存或使用虛擬內存

```bash
# 分批測試
python task_tester.py --tasks batch1.json --threads 2
python task_tester.py --tasks batch2.json --threads 2
```

#### 5. AgentFlow 生成時間過長

**問題**: AgentFlow 生成一個任務需要很長時間

**解決方案**:
- 減少 `--agentflow-turns` 參數
- 對簡單任務使用直接模式（不加 `--agentflow`）
- 檢查 API 響應時間

```bash
# 使用較少的輪次
python task_pipeline.py --num-tasks 10 --agentflow --agentflow-turns 3
```

### 調試技巧

#### 啟用詳細日誌

```bash
# 設置日誌級別為 DEBUG
export LOG_LEVEL=DEBUG

# 運行時啟用詳細輸出
python task_tester.py --tasks my_tasks.json --verbose
```

#### 使用模擬模式

```bash
# 不調用 LLM，快速測試流程
python task_tester.py --tasks my_tasks.json --dry-run --verbose
```

#### 檢查任務格式

```python
# 使用 Python 驗證任務格式
import json

with open('generated_tasks/my_tasks.json', 'r') as f:
    tasks = json.load(f)
    for i, task in enumerate(tasks):
        print(f"Task {i}:")
        print(f"  Query: {task['task']['q'][:50]}...")
        print(f"  Actions: {len(task['task']['agt'])}")
        print(f"  Valid: {task.get('validation_report', {}).get('valid', 'unknown')}")
```

### 獲取幫助

如遇到問題：

1. **查看日誌文件**: `task_tester.log`, `task_generator.log`
2. **啟用詳細模式**: 使用 `--verbose` 參數
3. **檢查文檔**: 閱讀 `AGENTFLOW_README.md` 和 `Compare_agentflow_direct.md`
4. **運行測試**: 使用 `--dry-run` 或 `--validate-only` 進行快速測試

---

# English Documentation

## Project Overview

**APIGen-MT-5k** is an advanced multi-turn task generation and testing system designed for customer service scenarios. The system implements the **AgentFlow** multi-turn iterative architecture, supports dual-model collaboration (GPT-4o + GPT-OSS-120B), and provides a complete pipeline for task generation, validation, testing, and analysis.

### 🎯 Key Features

- **🔄 AgentFlow Architecture**: Multi-turn iterative generation with Planner, Executor, Verifier, and Generator collaboration
- **🤖 Dual-Model System**: GPT-4o (user model) + GPT-OSS-120B (assistant model) collaboration
- **✅ Intelligent Validation**: Automatic validation of data consistency, user ID consistency, and tool invocation correctness
- **📊 Comprehensive Testing**: Parallel testing, multiple evaluation metrics, and result visualization
- **🔧 Flexible Configuration**: Centralized configuration management via `configs.py`
- **📈 Detailed Reports**: JSON, CSV, and visualization reports for in-depth analysis
- **🎓 Model Training**: Support for GPT-OSS 20B model reinforcement learning training (GRPO + SFT) for custom customer service task models

### 🏗️ System Architecture

*(Same architecture diagram as Chinese version)*

## Directory Structure

*(Same directory structure as Chinese version)*

## Quick Start

### 1. Install Dependencies

```bash
# Required dependencies
pip install openai

# Optional dependencies (for progress bars and visualizations)
pip install tqdm pandas matplotlib
```

### 2. Configure API Keys

Edit the `configs.py` file and set your API keys:

```python
@dataclass
class TauBenchConfig:
    # GPT-4o (user model)
    user_model: str = "gpt-4o"
    user_api_key: str = "your-gpt4o-api-key"
    user_base_url: str = "https://api.openai.com/v1"
    
    # GPT-OSS-120B (assistant model)
    default_model: str = "vllm-a40-gpt-oss-120b"
    default_api_key: str = "your-gpt-oss-api-key"
    default_base_url: str = "https://your-api-endpoint/api"
```

### 3. Basic Usage

```bash
# Generate tasks (AgentFlow mode)
python task_pipeline.py --num-tasks 10 --agentflow --output generated_tasks/my_tasks.json

# Test tasks (dual-model)
python task_tester.py --tasks generated_tasks/my_tasks.json --dual-model --verbose --save-results --visualize

# View results
ls results/
```

## Core Components

### 1. Task Tester (`task_tester.py`)

Task tester with dual-model support and multiple testing modes.

#### Main Features

- ✅ **Dual-Model Testing**: GPT-4o enhances queries + GPT-OSS-120B executes tasks
- ✅ **Parallel Testing**: Multi-threaded parallel processing for efficiency
- ✅ **Multiple Modes**: Real execution, simulated execution, validation-only
- ✅ **Detailed Metrics**: Precision, recall, F1 score, output match rate, etc.
- ✅ **Result Visualization**: Automatic chart and statistical report generation

#### Command Line Arguments

**Input Options**:
```bash
--tasks FILE                    # JSON file containing tasks to test
```

**Dual-Model Options**:
```bash
--dual-model                    # Enable dual-model approach
--enhance-query                 # Enhance query using user model
```

**Assistant Model Configuration** (GPT-OSS-120B):
```bash
--model MODEL                   # Model name
--api-key KEY                   # API key
--base-url URL                  # API base URL
```

**User Model Configuration** (GPT-4o):
```bash
--user-model MODEL              # User model name
--user-api-key KEY              # User model API key
--user-base-url URL             # User model API base URL
```

**Tool Configuration**:
```bash
--envs-path PATH                # Retail environment path (default: envs/retail)
```

**Execution Options**:
```bash
--threads N                     # Number of parallel threads (default: 1)
--verbose                       # Verbose output
--dry-run                       # Dry run (no LLM calls)
--validate-only                 # Validate tasks only (no model calls)
```

**Output Options**:
```bash
--output-dir DIR                # Results directory (default: results)
--save-results                  # Save detailed results
--visualize                     # Generate visualizations
```

#### Usage Examples

```bash
# Basic testing
python task_tester.py --tasks generated_tasks/my_tasks.json --verbose

# Dual-model testing with visualization
python task_tester.py \
    --tasks generated_tasks/my_tasks.json \
    --dual-model \
    --enhance-query \
    --threads 4 \
    --save-results \
    --visualize

# Validation only (no model calls)
python task_tester.py --tasks generated_tasks/my_tasks.json --validate-only

# Dry run (quick test)
python task_tester.py --tasks generated_tasks/my_tasks.json --dry-run --verbose
```

### 2. Task Generator (`task_generator.py`)

Task generator supporting both AgentFlow multi-turn generation and direct generation modes.

#### AgentFlow Architecture

AgentFlow is a multi-turn iterative generation architecture with the following components:

*(Same AgentFlow architecture diagram and component descriptions as Chinese version)*

#### Usage Examples

```python
from task_generator import TauBenchOpenAIGenerator

# Create generator
generator = TauBenchOpenAIGenerator("envs/retail")

# Generate single task with AgentFlow
result = generator.generate_task_with_agentflow(
    max_turns=5,
    include_metadata=True
)

# Batch generate tasks
tasks = generator.generate_diverse_tasks(
    num_tasks=10,
    use_agentflow=True
)

# Save tasks
generator.save_tasks_to_file(tasks, "generated_tasks/my_tasks.json")
```

For detailed AgentFlow documentation, see [AGENTFLOW_README.md](AGENTFLOW_README.md)

### 3. Task Pipeline (`task_pipeline.py`)

Task generation pipeline integrating generation, validation, review, and refinement.

#### Main Features

- 🔄 **Iterative Refinement**: Automatic iterations until tasks pass validation
- 👥 **Review Committee**: Multiple reviewers evaluate task quality
- 🔍 **User ID Validation**: Ensures consistent user IDs across all actions
- 📊 **Statistical Reports**: Detailed generation statistics and failure analysis
- 🎯 **Scenario Diversity**: Ensures generated tasks cover multiple scenario types

#### Command Line Arguments

```bash
--num-tasks N                   # Number of tasks to generate (default: 3)
--max-iterations N              # Max iterations per task (default: 3)
--output FILE                   # Output file path (default: generated_tasks/Sampled_Tasks.json)
--no-user-id-validation         # Disable user ID consistency validation
--committee-size N              # Review committee size (default: 3)
--agentflow                     # Use AgentFlow multi-turn generation
--agentflow-turns N             # Max turns for AgentFlow (default: 5)
```

#### Usage Examples

```bash
# Basic generation (direct mode)
python task_pipeline.py --num-tasks 10 --output generated_tasks/my_tasks.json

# Generate with AgentFlow
python task_pipeline.py --num-tasks 10 --agentflow --agentflow-turns 5 --output generated_tasks/agentflow_tasks.json

# Increase iterations and committee size
python task_pipeline.py --num-tasks 5 --max-iterations 5 --committee-size 5 --output generated_tasks/high_quality_tasks.json

# Disable user ID validation (use only in special cases)
python task_pipeline.py --num-tasks 10 --no-user-id-validation --output generated_tasks/tasks.json
```

#### Pipeline Flow

*(Same pipeline flow diagram as Chinese version)*

### 4. Configuration System (`configs.py`)

Centralized configuration management system.

#### Main Configuration Options

*(Same configuration options as Chinese version)*

## Usage Examples

### Complete Workflow

```bash
# Step 1: Generate high-quality tasks with AgentFlow
python task_pipeline.py \
    --num-tasks 20 \
    --agentflow \
    --agentflow-turns 5 \
    --max-iterations 3 \
    --committee-size 3 \
    --output generated_tasks/production_tasks.json

# Step 2: Test tasks with dual-model
python task_tester.py \
    --tasks generated_tasks/production_tasks.json \
    --dual-model \
    --enhance-query \
    --threads 8 \
    --save-results \
    --visualize \
    --output-dir results/production_test

# Step 3: View results
ls results/production_test/
# test_results_*.json      - Detailed results
# test_summary_*.json      - Summary report
# test_summary_*.csv       - CSV format
# test_visualization_*.png - Visualization charts
```

### Compare AgentFlow vs Direct Mode

```bash
# Use script to compare both methods
python scripts/compare_blueprint_agentflow.py \
    --num-tasks 10 \
    --direct-out generated_tasks/test_direct.json \
    --agent-out generated_tasks/test_agentflow.json \
    --agentflow-turns 5
```

For detailed comparison analysis, see [Compare_agentflow_direct.md](Compare_agentflow_direct.md)

### Analyze Data References

```bash
# Analyze real data references in tasks
python scripts/analyze_real_data_references.py \
    generated_tasks/my_tasks.json \
    --envs-path envs/retail \
    --output analysis_report.json
```

## Configuration System

### Modifying Configuration

You can modify configuration in the following ways:

1. **Directly edit `configs.py`** (recommended for permanent changes)
2. **Command-line argument overrides** (for temporary changes)
3. **Environment variables** (for sensitive information)

### Configuration Priority

```
Command-line arguments > configs.py settings > Default values
```

### Example: Custom Configuration

```python
# Use custom configuration in code
from configs import TauBenchConfig

config = TauBenchConfig()
config.num_tasks = 50
config.temperature = 0.7
config.max_workers = 4

# Use custom configuration
from task_pipeline import TaskConfigurationPipeline, PipelineConfig

pipeline_config = PipelineConfig(
    envs_path=config.envs_path,
    max_iterations=5,
    committee_size=5,
    use_agentflow=True
)

pipeline = TaskConfigurationPipeline(pipeline_config)
```

## Output Description

*(Same output descriptions as Chinese version)*

## Model Training & Testing

This project supports **GPT-OSS 20B** model training via reinforcement learning for customer service task execution. The training follows a two-phase approach:

### 📚 Complete Documentation

- **[模型訓練指南.md](模型訓練指南.md)** - Detailed model training tutorial (Chinese)
- **[模型測試指南.md](模型測試指南.md)** - Model deployment and testing guide (Chinese)

### 🛠️ Training Environment Setup

Multiple training environment options are available. Choose the one that fits your setup:

#### Local Environment (Recommended for Development)

```bash
# Standard installation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# For A100/H100 GPUs (better performance)
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

#### NVIDIA DGX Spark (For Large-Scale Training)

Train models up to 200B parameters (e.g., gpt-oss-120b):

```bash
# 1. Download and build Docker image
wget -O Dockerfile "https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark"
docker build -f Dockerfile -t unsloth-dgx-spark .

# 2. Launch container
docker run -it --gpus=all --net=host --ipc=host \
    -v $(pwd):$(pwd) -w $(pwd) unsloth-dgx-spark

# 3. Train inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Memory Requirements**:
- GPT-OSS 20B: 24-40 GB VRAM (4-bit quantization)
- GPT-OSS 120B: ~68 GB unified memory (QLoRA 4-bit)

**Resources**:
- 📚 [Unsloth DGX Spark Documentation](https://unsloth.ai/docs/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)
- 🐳 [DGX Spark Dockerfile](https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark)

#### Cloud Platforms

**Google Colab** (Free T4 GPU):
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**AWS/Azure/GCP** (A100/H100):
```bash
pip install "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

**See [模型訓練指南.md](模型訓練指南.md) for detailed setup instructions**

### 🎓 Training Pipeline Overview

#### Phase 1: SFT Pretraining (30-60 minutes)

Supervised fine-tuning to teach the model JSON format and basic tool selection:

```bash
# In project root directory
jupyter notebook gpt_oss_20b_tau_bench_rl_training.ipynb

# Execute Cell 1-16 (SFT Training)
```

**Features**:
- Uses 800 labeled examples
- Learns correct tool calling format
- JSON valid rate: ~95%
- Action accuracy: ~90%

#### Phase 2: GRPO Training (2-4 hours)

Reinforcement learning for policy optimization:

```bash
# Continue executing Cell 17-21 (GRPO Training)
```

**Reward Mechanism**:
- ✅ +1.0 per correct action name
- ✅ +0.5 per correct argument set
- ✅ +0.5 for correct ordering
- ❌ -0.3 per incorrect action
- ❌ -0.4 per missing required argument

**Expected Results**:
- Exact match rate: 70-85%
- Action F1 score: 75-90%
- Rewards increase over training

### 🚀 Model Deployment & Testing

#### 1. Deploy Model Service

After training, the model is saved in the project directory (default `outputs/` or custom path)

**Option A: Use FastAPI (Recommended)**

```bash
# Start model service
python server.py

# Service runs on http://localhost:8000
```

**Option B: Use vLLM (High Performance)**

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model outputs/merged_model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16
```

**Option C: Use Ollama (Simple Deployment)**

```bash
# Create Modelfile
echo 'FROM outputs/merged_model' > Modelfile.gpt-oss

# Create and run model
ollama create gpt-oss-20b-tau -f Modelfile.gpt-oss
ollama serve
```

#### 2. Configure Test Environment

Edit `configs.py` to point to local model service:

```python
@dataclass
class TauBenchConfig:
    # Use trained local model
    default_model: str = "gpt-oss-20b-tau"
    default_api_key: str = "EMPTY"  # Local service doesn't need key
    default_base_url: str = "http://localhost:8000/v1"
```

#### 3. Run Tests

```bash
# Test trained model
python task_tester.py \
    --tasks generated_tasks/my_tasks.json \
    --model gpt-oss-20b-tau \
    --base-url http://localhost:8000/v1 \
    --verbose \
    --save-results \
    --visualize

# View results
ls results/
```

### 📊 Performance Evaluation

After training, evaluate the model using these metrics:

```python
import json

# Load test results
with open('results/test_results_*.json', 'r') as f:
    results = json.load(f)

# Calculate success rate
total = len(results['results'])
success = sum(1 for r in results['results'] if r['reward'] >= 0.99)
print(f"Success Rate: {success/total:.2%}")
print(f"Avg Action F1: {results['summary']['avg_action_f1']:.2f}")
print(f"Avg Action Precision: {results['summary']['avg_action_precision']:.2f}")
print(f"Avg Action Recall: {results['summary']['avg_action_recall']:.2f}")
```

### 🔧 Training File Structure

```
APIGen-MT-5k/
├── gpt_oss_20b_tau_bench_rl_training.ipynb  # Main training notebook
├── dataset.jsonl                             # Training data (800 tasks)
├── 3.5k.jsonl                                # Extended dataset (3500+ tasks)
├── server.py                                 # FastAPI server
├── run_training.sh                           # Training script
├── dgx_train.slurm                           # SLURM configuration
├── training_configs.ini                      # Training configs
├── 模型訓練指南.md                            # Training guide
└── 模型測試指南.md                            # Testing guide

# After training (based on configuration):
outputs/
├── checkpoint-*/                             # Training checkpoints
├── final_model/                              # Final model (LoRA)
└── merged_model/                             # Merged complete model
```

### 💡 Training Recommendations

**GPU Requirements**:
- Recommended: A100 or H100 (40GB+ VRAM)
- Minimum: A40 or V100 (24GB VRAM)
- Training time: 4-6 hours (depending on GPU)

**Optimization Tips**:
1. **Increase data**: Use `3.5k.jsonl` (3500+ tasks) for better results
2. **Adjust LoRA rank**: Increase to 32 or 64 for more capacity
3. **Tune learning rate**: Try 1e-5, 5e-6, 2e-5
4. **Batch size**: Adjust `per_device_train_batch_size` based on VRAM

**Common Issues**:
- **CUDA out of memory**: Reduce batch size or sequence length
- **Training not converging**: Adjust learning rate or increase warm-up steps
- **Low accuracy (<60%)**: Increase training data or adjust reward function
- **Training too slow**: Use multi-GPU or reduce epochs

For more details, see:
- 📖 [模型訓練指南.md](模型訓練指南.md) - Complete training tutorial
- 📖 [模型測試指南.md](模型測試指南.md) - Deployment and testing guide
- 📊 [SUMMARY.md](SUMMARY.md) - Project overview and summary

## Troubleshooting

### Common Issues

#### 1. API Connection Failure

**Problem**: `Connection refused` or `API key invalid`

**Solution**:
- Check API keys and URLs in `configs.py`
- Verify network connection
- Confirm API service is running

```bash
# Test API connection
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.openai.com/v1/models
```

#### 2. Tool Execution Failure

**Problem**: `Tool 'xxx' not found` or `Tool execution failed`

**Solution**:
- Confirm `envs/retail/` directory exists
- Check tool implementations in `tools/` directory
- Verify data file formats (users.json, orders.json, products.json)

```bash
# Verify data files
python -c "import json; json.load(open('envs/retail/users.json'))"
```

#### 3. Task Validation Failure

**Problem**: `Validation failed` or `User not found`

**Solution**:
- Check if user IDs and order IDs in tasks exist in data files
- Run validation separately with `--validate-only`
- Review detailed validation reports

```bash
# Validate tasks only
python task_tester.py --tasks my_tasks.json --validate-only --verbose
```

#### 4. Out of Memory

**Problem**: `MemoryError` or system slowdown

**Solution**:
- Reduce `--threads` parameter value
- Process tasks in batches
- Increase system memory or use virtual memory

```bash
# Test in batches
python task_tester.py --tasks batch1.json --threads 2
python task_tester.py --tasks batch2.json --threads 2
```

#### 5. AgentFlow Generation Takes Too Long

**Problem**: AgentFlow takes a long time to generate a single task

**Solution**:
- Reduce `--agentflow-turns` parameter
- Use direct mode (without `--agentflow`) for simple tasks
- Check API response time

```bash
# Use fewer turns
python task_pipeline.py --num-tasks 10 --agentflow --agentflow-turns 3
```

### Debugging Tips

#### Enable Verbose Logging

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Enable verbose output at runtime
python task_tester.py --tasks my_tasks.json --verbose
```

#### Use Simulation Mode

```bash
# No LLM calls, use simulated responses
python task_tester.py --tasks my_tasks.json --dry-run --verbose
```

#### Check Task Format

```python
# Verify task format using Python
import json

with open('generated_tasks/my_tasks.json', 'r') as f:
    tasks = json.load(f)
    for i, task in enumerate(tasks):
        print(f"Task {i}:")
        print(f"  Query: {task['task']['q'][:50]}...")
        print(f"  Actions: {len(task['task']['agt'])}")
        print(f"  Valid: {task.get('validation_report', {}).get('valid', 'unknown')}")
```

### Getting Help

If you encounter issues:

1. **Check log files**: `task_tester.log`, `task_generator.log`
2. **Enable verbose mode**: Use `--verbose` parameter
3. **Read documentation**: 
   - Review `AGENTFLOW_README.md` and `Compare_agentflow_direct.md`
   - For model training: See [模型訓練指南.md](模型訓練指南.md)
   - For model testing: See [模型測試指南.md](模型測試指南.md)
4. **Run tests**: Use `--dry-run` or `--validate-only` for quick tests

---

## 📚 Additional Documentation

### Core System
- **[AGENTFLOW_README.md](AGENTFLOW_README.md)** - Detailed AgentFlow architecture documentation
- **[Compare_agentflow_direct.md](Compare_agentflow_direct.md)** - Comparison of generation methods
- **[SUMMARY.md](SUMMARY.md)** - Project overview and summary

### Model Training & Testing
- **[模型訓練指南.md](模型訓練指南.md)** - Complete GPT-OSS 20B training guide (Chinese)
- **[模型測試指南.md](模型測試指南.md)** - Model deployment and testing guide (Chinese)
- **[GPT_OSS_120B_Task_Performance_Report.md](GPT_OSS_120B_Task_Performance_Report.md)** - Performance analysis report

### Analysis Scripts
- **scripts/analysis_helpers.py** - Helper functions for analysis
- **scripts/analyze_generated_tasks.py** - Analyze generated task statistics
- **scripts/analyze_real_data_references.py** - Analyze real data references
- **scripts/compare_blueprint_agentflow.py** - Compare generation methods

---

## 🚀 Quick Links

| Task | Documentation | Script/File |
|------|--------------|-------------|
| Generate tasks | [Quick Start](#quick-start) | `task_pipeline.py` |
| Test tasks | [Task Tester](#1-task-tester-task_testerpy) | `task_tester.py` |
| Train model | [模型訓練指南.md](模型訓練指南.md) | `gpt_oss_20b_tau_bench_rl_training.ipynb` |
| Deploy model | [模型測試指南.md](模型測試指南.md) | `server.py` |
| Analyze results | [Analysis Scripts](#analysis-scripts) | `scripts/analyze_*.py` |
| Configure system | [Configuration System](#configuration-system) | `configs.py` |

---

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue in the repository.

---

**Last Updated**: 2026-01-28  
**Version**: 2.0.0  
**New in 2.0**: Model training support with GPT-OSS 20B (GRPO + SFT)
