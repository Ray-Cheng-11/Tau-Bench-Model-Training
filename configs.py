# config.py
"""
Tau-Bench 任務生成與驗證系統配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TauBenchConfig:
    """Tau-Bench 系統配置"""
    # 環境路徑
    envs_path: str = "envs/retail"
    
        # 模型配置
    user_model: str = os.getenv("USER_MODEL", "gpt-4o")
    user_api_key: str = os.getenv("USER_API_KEY", "")
    user_base_url: str = os.getenv("USER_BASE_URL", "https://api.openai.com/v1")
    default_model: str = os.getenv("DEFAULT_MODEL", "vllm-a40-gpt-oss-120b")
    default_api_key: str = os.getenv("DEFAULT_API_KEY", "")
    default_base_url: str = os.getenv("DEFAULT_BASE_URL", "https://openwebui-demo-61-222-206-87.sslip.io/api")
    
    # 生成配置
    num_tasks: int =1200
    temperature: float = 0.0
    max_tokens: int = 16384
    max_retries: int = 3
    retry_backoff_base: float = 1.5
    timeout: int = 180  # 3 分鐘
    max_instruction_display_length: int = 100
    
    # 驗證配置
    test_timeout: int = 180
    max_workers: int = 2
    
    # 輸出配置
    output_dir: str = "tau_bench_results"
    save_visualizations: bool = True
    
    # 質量指標閾值
    min_success_rate: float = 0.7
    min_action_recall: float = 0.6
    min_action_precision: float = 0.6

    write_keywords: list[str] = field(default_factory=lambda: ['cancel', 'modify', 'return', 'exchange', 'create', 'update', 'transfer'])

    user_target_actions: set[str] = field(default_factory=lambda: {
        'calculate',
        'get_order_details',
        'get_product_details',
        'get_user_details',
        'list_all_products_types',
        'exchange_delivered_order_items',
        'find_user_id_by_email',
        'find_user_id_by_name_zip',
        'get_order_details',
        'get_product_details',
        'get_user_details',
        'list_all_product_types',
        'modify_pending_order_address',
        'modify_pending_order_items',
        'modify_pending_order_payment',
        'modify_user_address',
        'return_delivered_order_items',
        'think',
        'transfer_to_human_agents',
    })
    # Scenario constants centralization
    scenario_keys: List[str] = field(default_factory=lambda: [
        'order_cancellation',
        'order_modification',
        'item_return',
        'item_exchange',
        'address_change',
        'payment_update',
        'order_inquiry',
        'product_inquiry',
    ])

    scenario_templates: Dict[str, str] = field(default_factory=lambda: {
        'order_cancellation': 'Customer needs to cancel their order',
        'order_modification': 'Customer wants to modify shipping address',
        'item_return': 'Customer wants to return delivered items',
        'item_exchange': 'Customer needs to exchange a product',
        'address_change': 'Customer wants to modify shipping address',
        'payment_update': 'Customer wants to update payment method',
        'order_inquiry': 'Customer inquires about order status',
        'product_inquiry': 'Customer asks about product availability',
    })

    # Map scenario keys to required or typical actions for detection
    scenario_action_map: Dict[str, List[str]] = field(default_factory=lambda: {
        'order_cancellation': ['cancel_pending_order', 'cancel_order'],
        'order_modification': ['modify_pending_order_items', 'modify_pending_order_address', 'modify_pending_order_payment'],
        'item_return': ['return_delivered_order_items'],
        'item_exchange': ['exchange_delivered_order_items'],
        'address_change': ['modify_user_address'],
        'payment_update': ['modify_user_payment'],
        'order_inquiry': ['get_order_details'],
        'product_inquiry': ['get_product_details', 'list_all_product_types'],
    })
    # Whether to retry generation if suggested_scenario doesn't match detected scenario
    scenario_match_retries: int = 3
    # If True, generator attempts to ensure final task matches the suggested scenario; if False, logs but accepts mismatch
    scenario_match_strict: bool = True
# 建立輸出目錄
os.makedirs("generated_tasks", exist_ok=True)
os.makedirs("results", exist_ok=True)