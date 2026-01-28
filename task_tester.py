# task_tester.py
import matplotlib.pyplot as plt
import argparse
import json
import logging
import os
import time
import re
import csv
import numpy as np

import datetime
import random

from typing import Dict, List, Any, Optional, Tuple, Callable, Counter
import copy
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import inspect
from tqdm import tqdm

from data_reader import TauBenchDataReader
from configs import TauBenchConfig
from task_generator import TaskValidator, ValidationReport


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose HTTP request logging from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call with execution results"""
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    inferred: bool = False  # Whether the tool name was inferred from arguments (parsing-only flag)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create a ToolCall from a dictionary"""
        return cls(
            name=data.get('name', ''),
            arguments=data.get('arguments', {}),
            result=data.get('result'),
            success=data.get('success', True),
            error=data.get('error'),
            execution_time=data.get('execution_time', 0.0),
            inferred=data.get('inferred', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'arguments': self.arguments,
            'result': self.result,
            'success': self.success,
            'error': self.error,
            'execution_time': self.execution_time,
            'inferred': self.inferred
        }


@dataclass
class Task:
    """ task representation supporting multiple input formats"""
    task_id: str
    query: str
    expected_actions: List[ToolCall]
    expected_outputs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], task_id: str = None) -> 'Task':
        """Create a Task from various dictionary formats"""
        # Handle generator output format
        if 'task' in data and isinstance(data['task'], dict):
            task_data = data['task']
            metadata = {
                'generation_metadata': data.get('metadata', {}),
                'validation_report': data.get('validation_report', {}),
                'corrections_applied': data.get('corrections_applied', {}),
                'success': data.get('success', True),
                'thought': data.get('thought', ''),
                'raw_response': data.get('raw_response', '')
            }
        else:
            task_data = data
            metadata = data.get('metadata', {})
        
        # Extract query
        query = task_data.get('q', '') or task_data.get('query', '')
        
        # Handle expected actions (support multiple common keys used in datasets)
        expected_actions = []
        actions_data = (
            task_data.get('agt', []) or
            task_data.get('expected_actions', []) or
            task_data.get('gold_actions', []) or
            task_data.get('expected') or
            []
        )
        for action_data in actions_data:
            if isinstance(action_data, dict):
                expected_actions.append(ToolCall.from_dict(action_data))
        
        # Handle expected outputs
        expected_outputs = task_data.get('ogt', []) or task_data.get('expected_outputs', [])
        if not isinstance(expected_outputs, list):
            expected_outputs = []
        
        # Generate task ID if not provided
        if not task_id:
            task_id = task_data.get('task_id') or f"task_{random.randint(1000, 9999)}"
        
        return cls(
            task_id=task_id,
            query=query,
            expected_actions=expected_actions,
            expected_outputs=expected_outputs,
            metadata=metadata
        )


@dataclass
class ModelResponse:
    """Represents a model's response to a task"""
    actions: List[ToolCall]
    reasoning: Optional[str] = None
    raw_response: str = ""
    parsing_success: bool = True
    parsing_error: Optional[str] = None
    execution_time: float = 0.0
    
    def get_action_names(self) -> List[str]:
        """Get list of action names in order"""
        return [action.name for action in self.actions]
    
    def get_successful_actions(self) -> List[ToolCall]:
        """Get only successfully executed actions"""
        return [action for action in self.actions if action.success]


@dataclass
class TestResult:
    """Comprehensive test result with all metrics"""
    task_id: str
    success: bool
    task: Task
    model_response: ModelResponse
    
    # Core metrics
    action_precision: float = 0.0  # % of model actions that were expected
    action_recall: float = 0.0     # % of expected actions model included
    action_f1: float = 0.0         # F1 score for action matching
    exact_action_match: bool = False  # Whether actions match exactly
    output_match_rate: float = 0.0    # % of expected outputs matched
    
    # Timing
    total_execution_time: float = 0.0
    model_response_time: float = 0.0
    tool_execution_time: float = 0.0
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Detailed analysis
    details: Dict[str, Any] = field(default_factory=dict)
    
    # APIGen-MT inspired quality metrics
    state_changes: Dict[str, Any] = field(default_factory=dict)  # Δ𝒮E tracking
    policy_compliance: Dict[str, Any] = field(default_factory=dict)  # Policy check results
    review_scores: Dict[str, float] = field(default_factory=dict)  # Committee review scores
    semantic_alignment: float = 0.0  # Alignment between query and actions


class ToolExecutor:
    """Real tool executor using retail environment data with state tracking"""
    
    def __init__(self, envs_path: str = "envs/retail", log_unmasked_fixed_args: bool = False):
        """
        Initialize tool executor with real tools only
        
        Args:
            envs_path: Path to the retail environment data
            log_unmasked_fixed_args: If True, retry logs include full (unmasked) fixed args at INFO level. Otherwise masked values are shown.
        """
        self.envs_path = Path(envs_path)
        self._tools = {}
        self._data_cache = None
        # Control how much detail to log about fixed args (masking for privacy by default)
        self.log_unmasked_fixed_args = bool(log_unmasked_fixed_args)
        
        # APIGen-MT inspired: State tracking for Δ𝒮E (environment state changes)
        self._initial_state = None
        self._current_state = None
        self._state_changes_log = []
        
        try:
            self.data_reader = TauBenchDataReader(str(envs_path))
            self._load_real_tools()
            logger.info(f"Initialized real tool executor with {len(self._tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load real tools: {e}")
            raise RuntimeError(f"Could not initialize real tool executor: {e}")
    
    def _load_real_tools(self):
        """Load real tool implementations from the retail environment"""
        try:
            # Try to import the tools package
            tools_module_name = "envs.retail.tools"
            tools_pkg = importlib.import_module(tools_module_name)
            
            # Get all tools
            all_tools = getattr(tools_pkg, "ALL_TOOLS", None)
            if not all_tools:
                raise ImportError("No ALL_TOOLS found in envs.retail.tools")
            
            for tool_cls in all_tools:
                try:
                    if hasattr(tool_cls, 'get_info'):
                        info = tool_cls.get_info()
                        function_name = info.get('function', {}).get('name')
                        if function_name:
                            self._tools[function_name] = {
                                'class': tool_cls,
                                'info': info,
                                'type': 'real'
                            }
                            logger.debug(f"Loaded tool: {function_name}")
                except Exception as e:
                    logger.debug(f"Failed to register tool {tool_cls}: {e}")
            
            if not self._tools:
                raise RuntimeError("No tools were successfully loaded")
                
        except Exception as e:
            logger.error(f"Real tool loading failed: {e}")
            raise
    

    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for the model"""
        return [tool_data['info'] for tool_data in self._tools.values()]
    
    def get_data(self) -> Dict[str, Any]:
        """Get retail data (cached)"""
        if self._data_cache is None and hasattr(self, 'data_reader'):
            self._data_cache = self.data_reader.read_data_files()
        return self._data_cache or {}
    
    def _fix_tool_arguments(self, tool_class, provided_args: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Fix argument mismatches between provided args and tool signature.
        
        Handles:
        1. Unexpected arguments (remove them)
        2. Missing required arguments (try to infer from provided args)
        3. Argument name variations (user_id vs order_id)
        
        Args:
            tool_class: The tool class to invoke
            provided_args: Arguments provided by the model
            tool_name: Name of the tool for logging
            
        Returns:
            Fixed arguments dict or None if unable to fix
        """
        try:
            # If the model wrapped arguments inside an 'arguments' key, unwrap it here
            if isinstance(provided_args, dict) and 'arguments' in provided_args and isinstance(provided_args['arguments'], dict):
                provided_args = provided_args['arguments']
                logger.debug(f"Unwrapped 'arguments' wrapper for {tool_name}")

            # Ensure provided_args is a dict
            if provided_args is None:
                provided_args = {}
            if not isinstance(provided_args, dict):
                logger.warning(f"Unexpected provided_args type for {tool_name}: {type(provided_args)}")
                return None

            sig = inspect.signature(tool_class.invoke)
            # Get expected parameters (exclude 'self' and 'data' if present)
            param_list = list(sig.parameters.keys())
            # Remove 'self' if present (instance methods)
            if param_list and param_list[0] == 'self':
                param_list = param_list[1:]
            # Remove 'data' if present (tools typically have signature data, ...)
            if param_list and param_list[0] == 'data':
                param_list = param_list[1:]
            expected_params = param_list

            # Get parameter details
            param_details = {p: sig.parameters[p] for p in expected_params}
            
            fixed_args = {}
            
            # Step 1: Add all matching arguments
            for key, value in (provided_args or {}).items():
                if key in expected_params:
                    fixed_args[key] = value
            
            # Step 2: Check for missing required arguments
            for param_name, param in param_details.items():
                if param_name in fixed_args:
                    continue  # Already have this
                
                # Check if this is a required parameter
                has_default = param.default != inspect.Parameter.empty
                
                if not has_default:
                    # Try to find a suitable substitute from provided args
                    found = False
                    
                    # Try exact semantic matches
                    if param_name == 'order_id' and 'orderId' in provided_args:
                        fixed_args[param_name] = provided_args['orderId']
                        found = True
                    elif param_name == 'order_id':
                        # Look for anything with 'order' and 'id' or common order keys first
                        order_keys = ['order_id', 'orderId', 'order', 'order_number', 'order_no', 'orderNo']
                        for k in order_keys:
                            if k in provided_args and provided_args[k]:
                                fixed_args[param_name] = provided_args[k]
                                found = True
                                break

                        # Fallback: look for keys containing order & id
                        if not found:
                            for key, value in provided_args.items():
                                if 'order' in key.lower() and ('id' in key.lower() or key.endswith('_id')):
                                    fixed_args[param_name] = value
                                    found = True
                                    break

                        # If still not found, attempt to infer from user/email by searching order data
                        if not found:
                            try:
                                data = self.get_data()
                                orders = data.get('orders', {})
                                users = data.get('users', {})

                                user_key = None
                                if 'user_id' in provided_args:
                                    user_key = provided_args['user_id']
                                elif 'userId' in provided_args:
                                    user_key = provided_args['userId']
                                elif 'email' in provided_args:
                                    for uid, u in users.items():
                                        if isinstance(u, dict) and u.get('email') == provided_args['email']:
                                            user_key = uid
                                            break

                                candidates = []
                                if user_key:
                                    for oid, od in orders.items():
                                        if od.get('user_id') == user_key:
                                            candidates.append((oid, od))

                                # Debug info about candidate orders for this user
                                logger.debug(f"Order inference: user_key={user_key}, candidates={[oid for oid,_ in candidates]}")

                                # Prefer pending orders
                                pending = [oid for oid, od in candidates if od.get('status') == 'pending']
                                if len(pending) == 1:
                                    fixed_args[param_name] = pending[0]
                                    found = True
                                elif len(pending) > 1:
                                    # If item information present, try to match by items
                                    provided_item_ids = set()
                                    # If we cannot unambiguously infer, expose candidates in debug and abort
                                    logger.debug(f"Multiple pending orders for user '{user_key}': {pending}")
                                    # We cannot safely infer which order to cancel without additional info
                                    logger.warning(f"Multiple pending orders found for user '{user_key}'; cannot infer 'order_id' automatically: {pending}")
                                    return None
                                    if 'item_ids' in provided_args and isinstance(provided_args['item_ids'], list):
                                        provided_item_ids = set(provided_args['item_ids'])
                                    elif 'items' in provided_args and isinstance(provided_args['items'], list):
                                        for it in provided_args['items']:
                                            if isinstance(it, dict) and 'item_id' in it:
                                                provided_item_ids.add(it['item_id'])
                                            elif isinstance(it, str):
                                                provided_item_ids.add(it)

                                    matches = []
                                    if provided_item_ids:
                                        for oid, od in candidates:
                                            order_item_ids = set()
                                            for it in od.get('items', []):
                                                if isinstance(it, dict) and 'item_id' in it:
                                                    order_item_ids.add(it['item_id'])
                                            if provided_item_ids.issubset(order_item_ids):
                                                matches.append(oid)
                                        if len(matches) == 1:
                                            fixed_args[param_name] = matches[0]
                                            found = True

                                # If not found but only one candidate exists overall, infer it (with info log)
                                if not found and len(candidates) == 1:
                                    fixed_args[param_name] = candidates[0][0]
                                    found = True
                                    logger.info(f"Inferred order_id '{fixed_args[param_name]}' from single order for user '{user_key}'")

                            except Exception as e:
                                logger.debug(f"Error inferring order_id: {e}")
                    
                    elif param_name == 'user_id':
                        # Look for anything with 'user' and 'id'
                        for key, value in provided_args.items():
                            if 'user' in key.lower() and ('id' in key.lower() or key.endswith('_id')):
                                fixed_args[param_name] = value
                                found = True
                                break

                    elif param_name == 'reason':
                        # Try to infer a cancellation reason from provided args or default to a safe value
                        reason_candidates = []

                        # Direct matches
                        for key, value in provided_args.items():
                            kl = key.lower()
                            if 'reason' in kl or 'cancel' in kl or 'note' in kl or 'message' in kl or 'comment' in kl:
                                if isinstance(value, str) and value.strip():
                                    reason_candidates.append(value.strip())

                        # Heuristic mapping: look for words indicating 'ordered by mistake'
                        mapped = None
                        for rc in reason_candidates:
                            low = rc.lower()
                            if 'mistake' in low or 'wrong' in low or 'accident' in low:
                                mapped = 'ordered by mistake'
                                break
                            if 'no longer' in low or 'not needed' in low or 'dont need' in low or 'dont want' in low:
                                mapped = 'no longer needed'
                                break

                        # If not found, pick safest default
                        if not mapped and reason_candidates:
                            mapped = reason_candidates[0]

                        if not mapped:
                            # Default to a conservative reason to allow cancellation to proceed
                            mapped = 'no longer needed'

                        fixed_args[param_name] = mapped
                        found = True

                    elif param_name == 'item_ids' and 'items' in provided_args:
                        # Try to extract item IDs from items list
                        items = provided_args['items']
                        if isinstance(items, list):
                            item_ids = []
                            for item in items:
                                if isinstance(item, dict) and 'item_id' in item:
                                    item_ids.append(item['item_id'])
                                elif isinstance(item, str):
                                    item_ids.append(item)
                            if item_ids:
                                fixed_args[param_name] = item_ids
                                found = True
                    
                    # Generic fallback: look for similar parameter names
                    if not found:
                        param_lower = param_name.lower()
                        for key, value in provided_args.items():
                            key_lower = key.lower()
                            # Check for substring match or similar naming
                            if param_lower in key_lower or key_lower in param_lower:
                                fixed_args[param_name] = value
                                found = True
                                break

                    # If still not found, this is a problem
                    if not found:
                        logger.warning(f"Cannot fix missing required argument '{param_name}' for {tool_name}")
                        # For certain parameters we can supply a safe default instead of failing
                        if param_name == 'reason':
                            # Default to a conservative cancellation reason
                            fixed_args[param_name] = 'no longer needed'
                            found = True
                            logger.info(f"Defaulted missing 'reason' to 'no longer needed' for {tool_name}")
                        elif param_name == 'order_id':
                            # Do not default order_id silently; it's too dangerous. Return None so the caller can handle.
                            return None
                        else:
                            return None
            
            # Debug: show expected parameter names
            logger.debug(f"Expected params for {tool_name}: {list(param_details.keys())}")
            # Final validation: ensure all required params are present
            missing_required = []
            for p_name, p in param_details.items():
                if p.default == inspect.Parameter.empty and p_name not in fixed_args:
                    missing_required.append(p_name)

            if missing_required:
                logger.warning(f"Cannot fix all required args for {tool_name}; missing: {missing_required}")
                return None

            logger.debug(f"Fixed arguments for {tool_name}: {list(fixed_args.keys())}")
            logger.debug(f"Full fixed args for {tool_name}: {fixed_args}")
            return fixed_args
            
        except Exception as e:
            logger.error(f"Error fixing arguments for {tool_name}: {e}")
            return None

    def _mask_value_for_logging(self, value, key_name: str = ''):
        """Mask potentially sensitive values for logs (emails, long strings, order IDs)"""
        try:
            if isinstance(value, str):
                kl = key_name.lower() if key_name else ''
                # Mask emails
                if 'email' in kl and '@' in value:
                    local, domain = value.split('@', 1)
                    if len(local) > 2:
                        return local[0] + '***' + local[-1] + '@' + domain
                    return '***@' + domain
                # Partially mask order ids
                if 'order' in kl and isinstance(value, str) and len(value) > 6:
                    return value[:2] + '...' + value[-2:]
                # Truncate very long strings
                if len(value) > 200:
                    return value[:100] + '...<truncated>'
                return value
            if isinstance(value, dict):
                return {k: self._mask_value_for_logging(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [self._mask_value_for_logging(v, key_name) for v in value]
            if isinstance(value, set):
                return [self._mask_value_for_logging(v, key_name) for v in sorted(value)]
            return value
        except Exception:
            return '<masked>'

    def _mask_args_for_logging(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(args, dict):
                return {}
            return {k: self._mask_value_for_logging(v, k) for k, v in args.items()}
        except Exception:
            return {}
    
    def execute_tool(self, tool_call: ToolCall) -> ToolCall:
        """Execute a single tool call using real tool implementations"""
        start_time = time.time()
        # Normalize tool name to handle channel annotations or commentary suffixes
        raw_name = tool_call.name
        # Remove common suffix patterns like '<|...|>commentary'
        try:
            sanitized = re.sub(r"<\|.*?\|>.*$", "", raw_name)  # 更強的清理
            sanitized = sanitized.split(":")[0].split(" ")[0]  # 移除 colon/space 後綴
            lookup_name = sanitized.strip().strip('"\'')
        except Exception:
            sanitized = raw_name
        # Trim and normalize spacing/casing
        sanitized = sanitized.strip()
        lookup_name = sanitized

        # Handle missing tool name explicitly to avoid confusing logs like "Tool  failed: Tool '' not found"
        if not lookup_name:
            tool_call.name = "<MISSING_TOOL_NAME>"
            tool_call.success = False
            tool_call.error = "Tool name missing in parsed action"
            tool_call.result = "Error: Missing tool name"
            tool_call.execution_time = time.time() - start_time
            logger.warning(f"Skipping tool execution: missing tool name in call (raw: {raw_name})")
            return tool_call

        if lookup_name not in self._tools:
            # Try case-insensitive exact match
            matches = [name for name in self._tools.keys() if name.lower() == lookup_name.lower()]
            if matches:
                suggestion = matches[0]
                tool_call.error = f"Tool '{tool_call.name}' not found. Did you mean '{suggestion}'?"
                tool_call.result = f"Error: Tool '{tool_call.name}' is not available. Suggested: '{suggestion}'"
                logger.warning(f"Tool '{lookup_name}' not found. Suggested match: '{suggestion}'")
            else:
                # Use fuzzy matching to find close tool names for diagnostic help
                import difflib
                close = difflib.get_close_matches(lookup_name, list(self._tools.keys()), n=3, cutoff=0.6)
                if close:
                    suggestion_str = ', '.join(close)
                    tool_call.error = f"Tool '{tool_call.name}' not found. Closest: {suggestion_str}"
                    tool_call.result = f"Error: Tool '{tool_call.name}' is not available. Closest matches: {suggestion_str}"
                    logger.warning(f"Tool '{lookup_name}' not found. Closest matches: {suggestion_str}")
                else:
                    tool_call.error = f"Tool '{tool_call.name}' not found"
                    tool_call.result = f"Error: Tool '{tool_call.name}' is not available"
                    logger.warning(f"Tool '{lookup_name}' not found and no close matches")

            tool_call.success = False
            tool_call.execution_time = time.time() - start_time
            return tool_call
        
        try:
            # Execute real tool
            tool_class = self._tools[lookup_name]['class']
            data = self.get_data()

            if hasattr(tool_class, 'invoke'):
                # Normalize tool_call.arguments before invocation: unwrap common 'arguments' wrapper, and ensure dict
                try:
                    if isinstance(tool_call.arguments, dict) and 'arguments' in tool_call.arguments and isinstance(tool_call.arguments['arguments'], dict):
                        logger.debug(f"Unwrapped 'arguments' wrapper for tool call '{lookup_name}' before invocation")
                        tool_call.arguments = tool_call.arguments['arguments']
                    elif tool_call.arguments is None:
                        tool_call.arguments = {}
                    elif not isinstance(tool_call.arguments, dict):
                        logger.warning(f"Tool call arguments for {lookup_name} were not a dict; resetting to empty dict. Received type: {type(tool_call.arguments)}")
                        tool_call.arguments = {}
                except Exception as e:
                    logger.debug(f"Error normalizing tool_call.arguments for {lookup_name}: {e}")

                try:
                    result = tool_class.invoke(data, **tool_call.arguments)
                    tool_call.result = str(result)
                    tool_call.success = True
                    # APIGen-MT: Record state changes for state-modifying operations
                    self._record_state_change(lookup_name, tool_call.arguments, tool_call.result)
                except TypeError as te:
                    # Handle TypeError from tool invocation by inspecting the invoke signature
                    msg = str(te)
                    logger.debug(f"Tool invocation TypeError for {lookup_name}: {msg}")
                    
                    # Try to fix argument mismatch
                    fixed_args = self._fix_tool_arguments(tool_class, tool_call.arguments, lookup_name)
                    
                    if fixed_args is not None:
                        try:
                            # INFO: show masked values by default; allow unmasked via configuration
                            if getattr(self, 'log_unmasked_fixed_args', False):
                                logger.info(f"Retrying {lookup_name} with fixed args: {fixed_args}")
                            else:
                                masked = self._mask_args_for_logging(fixed_args)
                                logger.info(f"Retrying {lookup_name} with fixed args: {masked}")

                            # Log full fixed args at debug level for diagnostics
                            logger.debug(f"Full fixed args for {lookup_name}: {fixed_args}")

                            result = tool_class.invoke(data, **fixed_args)
                            tool_call.result = str(result)
                            tool_call.success = True
                            # Update arguments to reflect what was actually used
                            tool_call.arguments = fixed_args
                            # APIGen-MT: Record state changes
                            self._record_state_change(lookup_name, fixed_args, tool_call.result)
                        except Exception as e2:
                            tool_call.success = False
                            tool_call.error = f"Argument fix failed: {e2}"
                            tool_call.result = f"Error: {e2}"
                            logger.error(f"Tool execution error after arg fix for {lookup_name}: {e2}")
                    else:
                        tool_call.success = False
                        tool_call.error = f"Could not fix arguments: {msg}"
                        tool_call.result = f"Error: {msg}"
                        logger.error(f"Could not fix arguments for {lookup_name}: {msg}")
                        
                except Exception as e:
                    tool_call.success = False
                    tool_call.error = str(e)
                    tool_call.result = f"Error executing {tool_call.name}: {str(e)}"
                    logger.error(f"Tool execution error for {tool_call.name}: {e}")
            else:
                raise AttributeError(f"Tool '{tool_call.name}' has no invoke method")

        except Exception as e:
            tool_call.success = False
            tool_call.error = str(e)
            tool_call.result = f"Error executing {tool_call.name}: {str(e)}"
            logger.error(f"Tool execution error for {tool_call.name}: {e}")
        
        tool_call.execution_time = time.time() - start_time
        return tool_call
    

    
    def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """Execute multiple tool calls in sequence"""
        executed_calls = []
        
        for call in tool_calls:
            executed_call = self.execute_tool(call)
            executed_calls.append(executed_call)
            
            # Log execution
            if executed_call.success:
                logger.debug(f"Tool {executed_call.name} executed successfully in {executed_call.execution_time:.3f}s")
            else:
                logger.warning(f"Tool {executed_call.name} failed: {executed_call.error}")
        
        return executed_calls
    
    def _record_state_change(self, tool_name: str, arguments: Dict[str, Any], result: str):
        """Record a state-changing action (APIGen-MT Δ𝒮E tracking)"""
        # Identify state-changing operations
        state_changing_tools = {
            'cancel_pending_order', 'modify_pending_order_address',
            'modify_pending_order_items', 'modify_pending_order_payment',
            'modify_user_address', 'return_delivered_order_items',
            'exchange_delivered_order_items'
        }
        
        if tool_name in state_changing_tools:
            change_entry = {
                'tool': tool_name,
                'arguments': arguments,
                'result': result,
                'timestamp': time.time()
            }
            self._state_changes_log.append(change_entry)


# Shared helpers to avoid duplicated implementations in multiple model clients
class BaseModelClient:
    """Base class providing shared helpers for model clients (tool execution, prompts, stop logic)."""

    def __init__(self, tool_executor: Optional[ToolExecutor] = None):
        self.tool_executor = tool_executor

    def _execute_and_add_results(self, actions: List[ToolCall], messages: List[Dict], tool_call_ids: List[str]) -> List[ToolCall]:
        """Execute tools and append results to messages using the configured tool executor."""
        if not self.tool_executor:
            logger.error("No tool executor available for executing actions")
            return actions

        executed_actions: List[ToolCall] = []
        for i, action in enumerate(actions):
            executed_action = self.tool_executor.execute_tool(action)
            executed_actions.append(executed_action)

            tool_call_id = tool_call_ids[i] if i < len(tool_call_ids) else f"call_{len(messages)}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": executed_action.result or str(executed_action.error)
            })

        return executed_actions

    def _should_stop_iteration(self, task: Task, all_actions: List[ToolCall]) -> bool:
        """Stop when we've covered a large fraction of expected actions (80%)."""
        expected_action_names = set(action.name for action in task.expected_actions)
        actual_action_names = set(action.name for action in all_actions)

        coverage = len(expected_action_names & actual_action_names) / len(expected_action_names) if expected_action_names else 1
        return coverage >= 0.8

    def _build_system_prompt(self) -> str:
        """Shared system prompt used by model clients."""
        # Strengthened system prompt aligned with task_generator guidance
        return (
            "You are a retail customer service assistant with access to tools to help customers.\n\n"
            "Follow a step-by-step approach and prefer SAFETY and DATA ACCURACY over creativity.\n\n"
            "Core approach:\n"
            "1) AUTHENTICATE first when the task implies modifications or accesses private data (call find_user_id_by_email or find_user_id_by_name_zip).\n"
            "2) GATHER information only as needed (get_user_details, get_order_details, get_product_details).\n"
            "3) EXECUTE modifications only after authentication of correct references.\n\n"
            "For ACTION requests (cancellations, modifications, returns, exchanges):\n"
            "- Follow: authenticate -> gather info -> perform action.\n"
            "- Use these modification tools when appropriate: cancel_pending_order, modify_pending_order_address, modify_pending_order_items,\n"
            "  modify_pending_order_payment, modify_user_address, return_delivered_order_items, exchange_delivered_order_items, transfer_to_human_agents.\n\n"
            "- When a user clearly requests a modification and identifiers are available (order ID, email), attempt the requested modification automatically.\n"
            "  Use tools to retrieve any missing identifiers (get_order_details, get_user_details) rather than asking the user for them.\n"
            "- Do not ask for extra confirmation before performing the requested modification unless the user explicitly requested confirmation or the operation requires additional consent (e.g., additional charges).\n"
            "- Only escalate via transfer_to_human_agents if the tool indicates an error, policy requires human review, or the operation cannot be completed with available tools.\n\n"
            "Specific guidance for order states:\n"
            "- If an order is in 'delivered' status and the user requests a return, refund or replacement, automatically call return_delivered_order_items to initiate the return and refund process, and optionally create a replacement order. Do not ask for user confirmation unless a payment change is needed.\n"
            "- If an order is 'processed' (i.e., fulfillment/tracking has been created), attempt the following in sequence: 1) try modify_pending_order_address, 2) try cancel_pending_order if address modify is not possible, 3) if both fail, call transfer_to_human_agents.\n"
            "- When performing returns or replacements, use the original payment method for refunds unless the user explicitly requests a different payment method.\n\n"
            "For INFORMATION requests (status checks, balance inquiries, product details):\n"
            "- Authenticate if the query includes personal data, otherwise call only information-gathering tools and return an answer.\n\n"
            "Mandatory constraints (CRITICAL):\n"
            "- ALWAYS use only the data present in the conversation and the task metadata. Do NOT invent user_ids, order_ids, product_ids, emails, or other data.\n"
            "- When a concrete identifier (order ID, email, user name+zip) is present in the task text, use it exactly.\n"
            "- Do NOT perform state-changing operations unless they directly satisfy the customer's stated intent. Avoid unnecessary changes.\n\n"
            "When producing tool calls, be precise: include required arguments and use the available tool names exactly. The system will provide the tool metadata to you."
        )

    def _build_user_prompt(self, task: Task, available_tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Shared user prompt builder.

        Adds available tool names and strict data usage guidance to align with the task generator prompts.
        """
        # Detect task type to provide appropriate guidance
        query_lower = task.query.lower()
        
        is_action_request = any(word in query_lower for word in [
            'cancel', 'modify', 'change', 'update', 'return', 'exchange', 
            'refund', 'replace', 'remove', 'add'
        ])
        
        is_info_request = any(word in query_lower for word in [
            'what', 'when', 'where', 'how much', 'status', 'balance', 
            'details', 'information', 'tell me', 'show me', 'check'
        ])
        
        if is_action_request:
            guidance = (
                "This is an ACTION request. Strictly follow: 1) authenticate -> 2) gather required info -> "
                "3) perform the minimal state-changing actions required.\n"
                "If the task intent clearly requests a modification and the identifiers are present or discoverable, perform the modification without asking for extra confirmation; only ask for confirmation when the user explicitly requests it or the modification requires extra consent.\n"
                "If the user explicitly asks for a refund/return/replacement for delivered items, start the return_delivered_order_items process immediately and issue refunds to the original payment method unless explicitly requested otherwise."
            )
        elif is_info_request and not is_action_request:
            guidance = (
                "This is an INFORMATION request. Authenticate only if necessary, call information-gathering tools, "
                "and return a concise, accurate answer based on tool results. Do not modify state."
            )
        else:
            guidance = (
                "Decide whether this requires action or only information. If action is required, authenticate first; otherwise, only gather info."
            )

        # Build available tools display to help the model (tool metadata is also provided by the API)
        tools_list = ''
        if available_tools:
            try:
                tool_names = [t.get('function', {}).get('name', t.get('name', '')) for t in available_tools]
                tools_list = '\nAvailable Tools: ' + ', '.join(tool_names)
            except Exception:
                tools_list = ''

        # Prefer explicit instruction to use only provided data and not invent anything
        data_constraints = (
            "CRITICAL: Use only information available in the customer request text and task metadata. "
            "Do NOT invent or hallucinate user IDs, order IDs, product IDs, emails, or other private data. "
            "If identifiers are missing, attempt to authenticate (find_user_id_by_email or find_user_id_by_name_zip) and use legitimate lookups."
        )

        # Include task metadata hints if available
        meta_hint = ''
        try:
            gen_meta = task.metadata.get('generation_metadata') if isinstance(task.metadata, dict) else None
            if gen_meta:
                # Keep short to avoid token bloat
                summary = []
                if isinstance(gen_meta, dict):
                    if gen_meta.get('scenario'):
                        summary.append(f"scenario={gen_meta.get('scenario')}")
                if summary:
                    meta_hint = '\nTask Metadata: ' + ', '.join(summary)
        except Exception:
            meta_hint = ''

        return (
            f"Customer Request: {task.query}\n\n"
            f"INSTRUCTIONS: {guidance}\n\n"
            f"{data_constraints}\n"
            f"{tools_list}\n"
            f"{meta_hint}\n\n"
            "Use the available tools to handle this request. Do not ask the user for additional information — use the tools to find necessary identifiers and proceed with requested modifications where appropriate."
        )


class ModelClient(BaseModelClient):
    """Handles communication with the GPT OSS model"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 tool_executor: ToolExecutor = None):
        """Initialize the model client"""
        # Initialize base with tool executor
        super().__init__(tool_executor)
        self.configs = TauBenchConfig()

        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', self.configs.default_api_key)
        self.base_url = base_url or os.environ.get('OPENAI_BASE_URL', self.configs.default_base_url)
        self.model = model or os.environ.get('OPENAI_MODEL', self.configs.default_model)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def get_model_response(self, task: Task, available_tools: List[Dict[str, Any]]) -> ModelResponse:
        """Get response from the model for a given task with iterative tool calling"""
        start_time = time.time()
        
        # Build initial prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task, available_tools)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        all_actions = []
        all_reasoning = []
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                
                # Make API call with function calling
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=10000,
                    timeout=self.configs.timeout
                )
                
                if not hasattr(response, 'choices') or not response.choices:
                    break
                    
                choice = response.choices[0]
                message = choice.message
                
                # Add assistant message to conversation
                assistant_message = {"role": "assistant"}
                if hasattr(message, 'content') and message.content:
                    assistant_message["content"] = message.content
                    all_reasoning.append(message.content)
                
                # Check for tool calls
                new_actions = []
                tool_call_ids = []
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    assistant_message["tool_calls"] = []
                    
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, 'function'):
                            function = tool_call.function
                            try:
                                arguments = json.loads(function.arguments) if function.arguments else {}
                                action = ToolCall(
                                    name=function.name,
                                    arguments=arguments
                                )
                                new_actions.append(action)
                                all_actions.append(action)
                                tool_call_ids.append(tool_call.id)
                                
                                # Add to assistant message
                                assistant_message["tool_calls"].append({
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": function.name,
                                        "arguments": function.arguments
                                    }
                                })
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse tool call arguments: {e}")
                
                messages.append(assistant_message)
                
                # If no new tools were called, we're done
                if not new_actions:
                    break
                
                # Execute the tools and add results to conversation
                executed_actions = self._execute_and_add_results(new_actions, messages, tool_call_ids)
                
                # Update the actions in all_actions with execution results
                for i, action in enumerate(new_actions):
                    if i < len(executed_actions):
                        action.result = executed_actions[i].result
                        action.success = executed_actions[i].success
                        action.error = executed_actions[i].error
                        action.execution_time = executed_actions[i].execution_time
                
                # Check if we should continue - if the task seems complete, stop
                if self._should_stop_iteration(task, all_actions):
                    break
            
            execution_time = time.time() - start_time
            
            # Combine all reasoning
            combined_reasoning = " ".join(all_reasoning) if all_reasoning else ""
            
            # Create raw response
            raw_response = json.dumps({
                "iterations": iteration,
                "reasoning": combined_reasoning,
                "tool_calls": [{"name": a.name, "arguments": a.arguments} for a in all_actions]
            }, indent=2)
            
            return ModelResponse(
                actions=all_actions,
                reasoning=combined_reasoning,
                raw_response=raw_response,
                parsing_success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"API call failed: {e}")
            
            return ModelResponse(
                actions=all_actions,
                reasoning="Error occurred during execution",
                raw_response=str(e),
                parsing_success=False,
                parsing_error=str(e),
                execution_time=execution_time
            )
    
    def _execute_and_add_results(self, actions: List[ToolCall], messages: List[Dict], tool_call_ids: List[str]) -> List[ToolCall]:
        return super()._execute_and_add_results(actions, messages, tool_call_ids)

    def _should_stop_iteration(self, task: Task, all_actions: List[ToolCall]) -> bool:
        """Determine if we should stop iterating based on task completion"""
        return super()._should_stop_iteration(task, all_actions)

    def _build_system_prompt(self) -> str:
        """Build system prompt for the model"""
        return super()._build_system_prompt()

    def _build_user_prompt(self, task: Task, available_tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build user prompt from task"""
        return super()._build_user_prompt(task, available_tools=available_tools)


class MockModelClient(BaseModelClient):
    """Lightweight mock client that returns the task's expected actions as if the model produced them.

    Useful for fast dry-runs and validating the generator output without calling LLMs or executing tools.
    """
    def __init__(self, tool_executor: ToolExecutor = None):
        super().__init__(tool_executor)

    def get_model_response(self, task: Task, available_tools: List[Dict[str, Any]]) -> ModelResponse:
        start_time = time.time()
        # Return a ModelResponse that mirrors the expected actions (no execution results)
        actions = []
        for exp in task.expected_actions:
            # Clone expected ToolCall but mark as not executed
            clone = ToolCall(
                name=exp.name,
                arguments=copy.deepcopy(exp.arguments),
                result=None,
                success=False,
                error="mock-not-executed",
                execution_time=0.0
            )
            actions.append(clone)

        execution_time = time.time() - start_time
        return ModelResponse(
            actions=actions,
            reasoning="[mock] returning expected actions",
            raw_response="[mock]",
            parsing_success=True,
            execution_time=execution_time
        )


class UserModelClient:
    """Handles communication with the User model (GPT-4o) for generating user queries"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None):

        self.configs = TauBenchConfig()
        """Initialize the user model client"""
        self.api_key = api_key or os.environ.get('USER_API_KEY', self.configs.user_api_key)
        self.base_url = base_url or os.environ.get('USER_BASE_URL', self.configs.user_base_url)
        self.model = model or os.environ.get('USER_MODEL', self.configs.user_model)

        # Initialize OpenAI client for GPT-4o
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"Initialized User Model Client with model: {self.model}")
    
    def generate_user_query(self, scenario: str) -> str:
        """Generate a realistic user query based on a scenario"""
        system_prompt = """You are a customer who needs help with retail/e-commerce issues. 
Generate realistic, natural customer service requests based on the given scenario.
Make the request sound like a real person would write it - include relevant details, emotions, and context.
Keep it conversational and authentic."""
        
        user_prompt = f"Create a customer service request for this scenario: {scenario}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                timeout=self.configs.timeout
            )

            # Robust content extraction (handles object or dict responses)
            content = None
            try:
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                    elif isinstance(choice, dict):
                        content = choice.get('message', {}).get('content')
                elif isinstance(response, dict):
                    choices = response.get('choices', [])
                    if choices:
                        content = choices[0].get('message', {}).get('content')
            except Exception:
                content = None

            if content:
                return content.strip()
            else:
                return f"Default request: {scenario}"

        except Exception as e:
            logger.error(f"User model query generation failed: {e}")
            return f"Default request: {scenario}"


class AssistantModelClient(BaseModelClient):
    """Handles communication with the Assistant model (GPT-OSS-120b) for tool calling"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 tool_executor: ToolExecutor = None):

        self.configs = TauBenchConfig()
        """Initialize the assistant model client"""
        # Initialize base with tool executor
        super().__init__(tool_executor)

        self.api_key = api_key or os.environ.get('ASSISTANT_API_KEY', self.configs.default_api_key)
        self.base_url = base_url or os.environ.get('ASSISTANT_BASE_URL', self.configs.default_base_url)
        self.model = model or os.environ.get('ASSISTANT_MODEL', self.configs.default_model)

        # Initialize OpenAI client for GPT-OSS-120b
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"Initialized Assistant Model Client with model: {self.model}")
    
    def get_model_response(self, task: Task, available_tools: List[Dict[str, Any]]) -> ModelResponse:
        """Get response from the assistant model for a given task with iterative tool calling"""
        start_time = time.time()
        # Build initial prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task, available_tools)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        all_actions = []
        all_reasoning = []
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                
                # Make API call with function calling
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=10000,
                    timeout=self.configs.timeout
                )
                
                if not hasattr(response, 'choices') or not response.choices:
                    break
                    
                choice = response.choices[0]
                message = choice.message
                
                # Add assistant message to conversation
                assistant_message = {"role": "assistant"}
                if hasattr(message, 'content') and message.content:
                    assistant_message["content"] = message.content
                    all_reasoning.append(message.content)
                
                # Check for tool calls
                new_actions = []
                tool_call_ids = []
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    assistant_message["tool_calls"] = []
                    
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, 'function'):
                            function = tool_call.function
                            try:
                                arguments = json.loads(function.arguments) if function.arguments else {}
                                action = ToolCall(
                                    name=function.name,
                                    arguments=arguments
                                )
                                new_actions.append(action)
                                all_actions.append(action)
                                tool_call_ids.append(tool_call.id)
                                
                                # Add to assistant message
                                assistant_message["tool_calls"].append({
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": function.name,
                                        "arguments": function.arguments
                                    }
                                })
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse tool call arguments: {e}")
                
                messages.append(assistant_message)
                
                # If no new tools were called, we're done
                if not new_actions:
                    break
                
                # Execute the tools and add results to conversation
                executed_actions = self._execute_and_add_results(new_actions, messages, tool_call_ids)
                
                # Update the actions in all_actions with execution results
                for i, action in enumerate(new_actions):
                    if i < len(executed_actions):
                        action.result = executed_actions[i].result
                        action.success = executed_actions[i].success
                        action.error = executed_actions[i].error
                        action.execution_time = executed_actions[i].execution_time
                
                # Check if we should continue - if the task seems complete, stop
                if self._should_stop_iteration(task, all_actions):
                    break
            
            execution_time = time.time() - start_time
            
            # Combine all reasoning
            combined_reasoning = " ".join(all_reasoning) if all_reasoning else ""
            
            # Create raw response
            raw_response = json.dumps({
                "iterations": iteration,
                "reasoning": combined_reasoning,
                "tool_calls": [{"name": a.name, "arguments": a.arguments} for a in all_actions]
            }, indent=2)
            
            return ModelResponse(
                actions=all_actions,
                reasoning=combined_reasoning,
                raw_response=raw_response,
                parsing_success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Assistant API call failed: {e}")
            
            return ModelResponse(
                actions=all_actions,
                reasoning="Error occurred during execution",
                raw_response=str(e),
                parsing_success=False,
                parsing_error=str(e),
                execution_time=execution_time
            )
    
    def _execute_and_add_results(self, actions: List[ToolCall], messages: List[Dict], tool_call_ids: List[str]) -> List[ToolCall]:
        return super()._execute_and_add_results(actions, messages, tool_call_ids)

    def _should_stop_iteration(self, task: Task, all_actions: List[ToolCall]) -> bool:
        """Determine if we should stop iterating based on task completion"""
        return super()._should_stop_iteration(task, all_actions)

    def _build_system_prompt(self) -> str:
        """Build system prompt for the assistant model"""
        return super()._build_system_prompt()

    def _build_user_prompt(self, task: Task, available_tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build user prompt from task"""
        return super()._build_user_prompt(task, available_tools=available_tools)


class DualModelClient:
    """Orchestrates interaction between User model (GPT-4o) and Assistant model (GPT-OSS-120b)"""
    
    def __init__(self,
                 user_api_key: str = None,
                 user_base_url: str = None,
                 user_model: str = None,
                 assistant_api_key: str = None,
                 assistant_base_url: str = None,
                 assistant_model: str = None,
                 tool_executor: ToolExecutor = None):
        """Initialize the dual model client"""
        
        self.user_client = UserModelClient(user_api_key, user_base_url, user_model)
        self.assistant_client = AssistantModelClient(assistant_api_key, assistant_base_url, assistant_model, tool_executor)
        
        logger.info(f"Initialized Dual Model Client - User: {self.user_client.model}, Assistant: {self.assistant_client.model}")
    
    def get_model_response(self, task: Task, available_tools: List[Dict[str, Any]]) -> ModelResponse:
        """Get response using dual-model approach"""
        # Use the assistant model to handle the task with tools
        return self.assistant_client.get_model_response(task, available_tools)
    
    def enhance_task_with_user_model(self, task: Task) -> Task:
        """Enhance task query using the user model for more realistic queries"""
        if task.query and len(task.query.strip()) > 0:
            # Generate a more natural user query based on the original
            query = self.user_client.generate_user_query(task.query)
            
            # Create enhanced task
            task = Task(
                task_id=task.task_id,
                query=query,
                expected_actions=task.expected_actions,
                expected_outputs=task.expected_outputs,
                metadata={**task.metadata, 'original_query': task.query, 'by_user_model': True}
            )
            
            return task


class ReviewCommittee:
    """
    APIGen-MT inspired: Committee of LLM reviewers for quality assessment
    Implements majority voting for stable, unbiased evaluation
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, num_reviewers: int = 3):
        """
        Initialize review committee
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI base URL
            model: Model to use for reviewers (default: gpt-4o)
            num_reviewers: Number of reviewers in committee (default: 3)
        """
        self.configs = TauBenchConfig()
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', self.configs.default_api_key)
        self.base_url = base_url or os.environ.get('OPENAI_BASE_URL', self.configs.default_base_url)
        self.model = model or 'gpt-4o'
        self.num_reviewers = num_reviewers
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        logger.info(f"Initialized Review Committee with {num_reviewers} reviewers using {self.model}")
    
    def review_task_quality(self, task: Task, model_response: ModelResponse, 
                           state_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review task execution quality using committee of LLM judges
        
        Metrics (from APIGen-MT paper):
        - Correctness: Do actions achieve the stated intent?
        - Completeness: Are all necessary steps included?
        - Satisfaction: Does it meet user expectations?
        - Creativity: Is the solution appropriate and efficient?
        
        return (
            Dict with scores and feedback from committee
        """
        
        # Build review prompt
        review_prompt = self._build_review_prompt(task, model_response, state_changes)
        
        # Collect reviews from committee
        reviews = []
        for i in range(self.num_reviewers):
            try:
                review = self._get_single_review(review_prompt, reviewer_id=i)
                reviews.append(review)
            except Exception as e:
                logger.warning(f"Reviewer {i} failed: {e}")
        
        if not reviews:
            logger.warning("No reviews collected from committee")
            return self._default_review()
        
        # Apply majority voting and aggregate
        aggregated_scores = self._aggregate_reviews(reviews)
        
        return aggregated_scores
    
    def _build_review_prompt(self, task: Task, model_response: ModelResponse, 
                            state_changes: Dict[str, Any]) -> str:
        """Build comprehensive review prompt"""
        
        expected_actions_str = "\n".join([
            f"  {i+1}. {action.name}({json.dumps(action.arguments)})"
            for i, action in enumerate(task.expected_actions)
        ])
        
        actual_actions_str = "\n".join([
            f"  {i+1}. {action.name}({json.dumps(action.arguments)}) -> {'✓' if action.success else '✗'}"
            for i, action in enumerate(model_response.actions)
        ])
        
        state_changes_str = json.dumps(state_changes, indent=2)
        
        prompt = f"""You are an expert evaluator assessing the quality of an AI agent's task execution.

**Task Query:** {task.query}

**Expected Actions (Ground Truth):**
{expected_actions_str}

**Actual Actions Taken:**
{actual_actions_str}

**Environment State Changes:**
{state_changes_str}

**Expected Outputs:** {', '.join(task.expected_outputs) if task.expected_outputs else 'None specified'}

Please evaluate the agent's performance on the following metrics (scale 0-10):

1. **Correctness**: Do the actions correctly achieve the user's stated intent?
2. **Completeness**: Are all necessary steps included? Nothing missing?
3. **Satisfaction**: Would this solution satisfy a real user?
4. **Efficiency**: Is the solution efficient (no unnecessary actions)?

Provide your evaluation in JSON format:
{{
    "correctness": <score 0-10>,
    "completeness": <score 0-10>,
    "satisfaction": <score 0-10>,
    "efficiency": <score 0-10>,
    "overall": <average score>,
    "reasoning": "<brief explanation of your assessment>",
    "issues": ["<list any problems>"],
    "strengths": ["<list what was done well>"]
}}

Be objective and consider semantic equivalence (e.g., different actions may achieve the same goal).
"""
        return prompt
    
    def _get_single_review(self, prompt: str, reviewer_id: int) -> Dict[str, Any]:
        """Get review from a single reviewer"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI agent evaluator. Provide objective, fair assessments."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                timeout=30
            )
            
            content = response.choices[0].message.content
            review = json.loads(content)
            review['reviewer_id'] = reviewer_id
            return review
            
        except Exception as e:
            logger.error(f"Review failed: {e}")
            raise
    
    def _aggregate_reviews(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate reviews using majority voting and averaging"""
        
        if not reviews:
            return self._default_review()
        
        # Average numerical scores
        metrics = ['correctness', 'completeness', 'satisfaction', 'efficiency']
        aggregated = {}
        
        for metric in metrics:
            scores = [r.get(metric, 0) for r in reviews if metric in r]
            aggregated[metric] = np.mean(scores) if scores else 0
        
        aggregated['overall'] = np.mean([aggregated[m] for m in metrics])
        
        # Collect all feedback
        all_issues = []
        all_strengths = []
        all_reasoning = []
        
        for review in reviews:
            all_issues.extend(review.get('issues', []))
            all_strengths.extend(review.get('strengths', []))
            all_reasoning.append(review.get('reasoning', ''))
        
        aggregated['issues'] = list(set(all_issues))  # Deduplicate
        aggregated['strengths'] = list(set(all_strengths))
        aggregated['reasoning'] = ' | '.join(all_reasoning)
        aggregated['num_reviewers'] = len(reviews)
        
        return aggregated
    
    def _default_review(self) -> Dict[str, Any]:
        """Return default review when committee fails"""
        return {
            'correctness': 0,
            'completeness': 0,
            'satisfaction': 0,
            'efficiency': 0,
            'overall': 0,
            'reasoning': 'Review committee failed',
            'issues': ['Unable to collect reviews'],
            'strengths': [],
            'num_reviewers': 0
        }


class EvaluationEngine:
    """Comprehensive evaluation engine optimized with Tau-Bench research insights"""
    
    def __init__(self, enable_review_committee: bool = False):
        """
        Initialize evaluation engine
        
        Args:
            enable_review_committee: Whether to use LLM review committee (APIGen-MT feature)
        """
        # Expanded equivalence groups based on task generation guidelines
        self.action_equivalence_groups = [
            # Authentication equivalence - both achieve user identification
            {'find_user_id_by_email', 'find_user_id_by_name_zip'},
            # Address modification equivalence - context-dependent  
            {'modify_pending_order_address', 'modify_user_address'},
            # Order state change equivalence - all change order status
            {'cancel_pending_order', 'return_delivered_order_items', 'exchange_delivered_order_items'},
            # Information retrieval equivalence - all gather customer data
            {'get_user_details', 'get_order_details', 'get_product_details'},
            # Item modification equivalence
            {'modify_pending_order_items', 'exchange_delivered_order_items'},
            # Payment handling equivalence
            {'modify_pending_order_payment', 'return_delivered_order_items'}  # refund scenarios
        ]
        
        # Action categories for logical sequence analysis
        self.action_categories = {
            'authentication': {'find_user_id_by_email', 'find_user_id_by_name_zip'},
            'information': {'get_user_details', 'get_order_details', 'get_product_details', 
                          'list_all_product_types', 'get_user_order_history'},
            'modification': {'cancel_pending_order', 'modify_pending_order_address', 
                           'modify_pending_order_items', 'modify_pending_order_payment',
                           'modify_user_address', 'return_delivered_order_items', 
                           'exchange_delivered_order_items'},
            'complaint_escalation': {'transfer_to_human_agents'},
            'communication': {'send_message'}  # For non-tool outputs
        }
        
        # Task type patterns
        self.action_keywords = {
            'cancel', 'modify', 'change', 'update', 'return', 'exchange', 
            'refund', 'replace', 'remove', 'add', 'transfer'
        }
        self.info_keywords = {
            'what', 'when', 'where', 'how much', 'status', 'balance', 
            'details', 'information', 'tell me', 'show me', 'check', 'find'
        }
        
        self.quality_metrics_history = []
        
        # APIGen-MT: Optional review committee for quality assessment
        self.enable_review_committee = enable_review_committee
        self.review_committee = ReviewCommittee() if enable_review_committee else None

    def evaluate(self, task: Task, model_response: ModelResponse, 
                total_execution_time: float, state_changes: Dict[str, Any] = None) -> TestResult:
        """Evaluation with multi-dimensional quality assessment (APIGen-MT enhanced)"""
        
        # Detect task type for context-aware evaluation
        task_type = self._detect_task_type(task.query)
        
        # Extract action names for comparison
        expected_action_names = [action.name for action in task.expected_actions]
        actual_action_names = [action.name for action in model_response.actions]

        # Calculate action metrics with sequence awareness and task type context
        metrics = self._calculate_action_metrics(
            expected_action_names, 
            actual_action_names,
            task_type
        )

        # Calculate semantic output matching
        output_match_rate = self._calculate_semantic_output_match(
            task.expected_outputs, 
            self._extract_outputs(model_response)
        )
        
        # Success determination with quality thresholds and task type awareness
        success = self._determine_success(task, model_response, metrics, task_type)
        
        # Calculate comprehensive timing analysis
        timing_metrics = self._calculate_timing_metrics(model_response, total_execution_time)
        
        # Generate analysis with quality insights
        details = self._generate_analysis(task, model_response, metrics, timing_metrics, task_type)

        # Generate intelligent warnings with improvement suggestions
        warnings = self._generate_intelligent_warnings(task, model_response, metrics, task_type)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(metrics, output_match_rate, timing_metrics)
        
        # APIGen-MT: Get review committee scores if enabled
        review_scores = {}
        semantic_alignment = 0.0
        if self.enable_review_committee and self.review_committee and state_changes:
            try:
                review_scores = self.review_committee.review_task_quality(task, model_response, state_changes)
                semantic_alignment = review_scores.get('overall', 0.0) / 10.0  # Normalize to 0-1
                logger.info(f"Review Committee Score: {review_scores.get('overall', 0):.2f}/10")
            except Exception as e:
                logger.warning(f"Review committee failed: {e}")
        
        result = TestResult(
            task_id=task.task_id,
            success=success,
            task=task,
            model_response=model_response,
            action_precision=metrics['precision'],
            action_recall=metrics['recall'],
            action_f1=metrics['f1'],
            exact_action_match=metrics['exact_match'],
            output_match_rate=output_match_rate,
            total_execution_time=total_execution_time,
            model_response_time=model_response.execution_time,
            tool_execution_time=timing_metrics['tool_execution_time'],
            details=details,
            warnings=warnings,
            # APIGen-MT enhancements
            state_changes=state_changes or {},
            review_scores=review_scores,
            semantic_alignment=semantic_alignment
        )
        
        # Add metrics
        result.details.update({
            'task_type': task_type,
            'quality_score': quality_score,
            'quality_rating': self._get_quality_rating(quality_score),
            'sequence_quality': metrics['sequence_quality'],
            'efficiency_metrics': timing_metrics,
            'improvement_suggestions': self._generate_improvement_suggestions(metrics, timing_metrics, task_type)
        })
        
        return result
    
    def _detect_task_type(self, query: str) -> str:
        """Detect task type: 'action', 'information', or 'mixed'"""
        query_lower = query.lower()
        
        has_action = any(word in query_lower for word in self.action_keywords)
        has_info = any(word in query_lower for word in self.info_keywords)
        
        if has_action and has_info:
            return 'mixed'
        elif has_action:
            return 'action'
        elif has_info:
            return 'information'
        else:
            return 'unknown'

    def _calculate_action_metrics(self, expected: List[str], actual: List[str], task_type: str = 'unknown') -> Dict[str, float]:
        """Calculate comprehensive action metrics with sequence awareness and task type context"""
        if not expected and not actual:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'exact_match': True, 'sequence_quality': 1.0}
        
        # Basic set-based metrics
        expected_set = set(expected)
        actual_set = set(actual)
        
        # Matching with semantic equivalence
        matched_expected, matched_actual = self._calculate_semantic_matches(expected_set, actual_set)
        
        # Calculate precision and recall with equivalence consideration
        precision = len(matched_actual) / len(actual_set) if actual_set else 0.0
        recall = len(matched_expected) / len(expected_set) if expected_set else 1.0 if not actual_set else 0.0
        
        # F1 calculation
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match with order consideration
        exact_match = expected == actual
        
        # Sequence quality assessment
        sequence_quality = self._calculate_sequence_quality(expected, actual)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact_match,
            'sequence_quality': sequence_quality
        }

    def _calculate_semantic_matches(self, expected: set, actual: set) -> tuple:
        """Calculate matches considering semantic equivalence and action categories"""
        matched_expected = set()
        matched_actual = set()
        
        # Direct matches first
        direct_matches = expected & actual
        matched_expected.update(direct_matches)
        matched_actual.update(direct_matches)
        
        # Semantic equivalence matches
        remaining_expected = expected - matched_expected
        remaining_actual = actual - matched_actual
        
        for exp_action in list(remaining_expected):
            equivalent_found = False
            for group in self.action_equivalence_groups:
                if exp_action in group:
                    # Find equivalent actions in remaining actual
                    equivalent_matches = group & remaining_actual
                    if equivalent_matches:
                        matched_action = next(iter(equivalent_matches))
                        matched_expected.add(exp_action)
                        matched_actual.add(matched_action)
                        remaining_expected.discard(exp_action)
                        remaining_actual.discard(matched_action)
                        equivalent_found = True
                        break
            
            # Category-level matching for remaining actions
            if not equivalent_found:
                exp_category = self._get_action_category(exp_action)
                if exp_category:
                    category_matches = {act for act in remaining_actual 
                                      if self._get_action_category(act) == exp_category}
                    if category_matches:
                        matched_action = next(iter(category_matches))
                        matched_expected.add(exp_action)
                        matched_actual.add(matched_action)
                        remaining_expected.discard(exp_action)
                        remaining_actual.discard(matched_action)
        
        return matched_expected, matched_actual

    def _calculate_sequence_quality(self, expected: List[str], actual: List[str]) -> float:
        """Calculate sequence quality using multiple alignment strategies"""
        if not expected or not actual:
            return 0.5  # Neutral score for empty sequences
        
        # Longest Common Subsequence for order preservation
        lcs_score = self._calculate_lcs_similarity(expected, actual)
        
        # Category sequence alignment
        category_score = self._calculate_category_alignment(expected, actual)
        
        # Logical flow assessment
        logical_score = self._assess_logical_flow(actual)
        
        # Weighted combination
        return 0.4 * lcs_score + 0.3 * category_score + 0.3 * logical_score

    def _calculate_lcs_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate Longest Common Subsequence similarity"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n) if max(m, n) > 0 else 0.0

    def _calculate_category_alignment(self, expected: List[str], actual: List[str]) -> float:
        """Calculate alignment based on action categories"""
        expected_categories = [self._get_action_category(action) for action in expected]
        actual_categories = [self._get_action_category(action) for action in actual]
        
        return self._calculate_lcs_similarity(expected_categories, actual_categories)

    def _assess_logical_flow(self, actions: List[str]) -> float:
        """Assess logical flow of action sequence with permissive pattern matching"""
        if len(actions) <= 1:
            return 1.0
        
        categories = [self._get_action_category(action) for action in actions]
        
        # Define valid logical patterns based on business rules (expanded and more flexible)
        valid_patterns = [
            ['authentication', 'information', 'modification'],  # Standard flow
            ['information', 'authentication', 'modification'],  # Info first flow
            ['authentication', 'modification'],  # Direct modification
            ['information', 'modification'],  # Info then modify
            ['authentication', 'information'],  # Auth then info (info requests)
            ['information', 'information'],  # Pure information request
            ['authentication', 'information', 'information'],  # Auth + multiple info
            ['authentication', 'complaint_escalation'],  # Authentication then escalation
            ['information', 'complaint_escalation'],  # Info then escalation
            ['authentication', 'information', 'modification', 'information'],  # Complex flow
            ['authentication', 'information', 'information', 'modification'],  # Multi-info then modify
        ]
        
        # Check for exact pattern match
        for pattern in valid_patterns:
            if categories == pattern:
                return 1.0
        
        # Check for subsequence match (allows extra steps)
        for pattern in valid_patterns:
            if self._is_subsequence(pattern, categories):
                return 0.9  # Slightly lower for extra steps
        
        # Partial pattern matching with more lenient threshold
        for pattern in valid_patterns:
            if len(categories) == len(pattern):
                matches = sum(1 for cat1, cat2 in zip(categories, pattern) if cat1 == cat2)
                match_ratio = matches / len(pattern)
                if match_ratio >= 0.6:  # Reduced from 0.7
                    return 0.6 + (match_ratio - 0.6) * 1.0  # Scale 0.6-1.0 to 0.6-1.0
        
        # Check for valid category transitions (no invalid jumps)
        valid_transitions = {
            'authentication': {'information', 'modification', 'complaint_escalation', 'authentication'},
            'information': {'information', 'modification', 'complaint_escalation', 'authentication'},
            'modification': {'information', 'modification', 'complaint_escalation'},
            'complaint_escalation': {'complaint_escalation'},
            'communication': {'communication', 'complaint_escalation'},
            'other': {'authentication', 'information', 'modification', 'complaint_escalation', 'other'}
        }
        
        transition_score = 0.0
        for i in range(len(categories) - 1):
            current = categories[i]
            next_cat = categories[i + 1]
            if next_cat in valid_transitions.get(current, set()):
                transition_score += 1
        
        if len(categories) > 1:
            transition_ratio = transition_score / (len(categories) - 1)
            if transition_ratio >= 0.8:
                return max(0.5, transition_ratio)  # Good transitions = decent score
        
        return 0.4  # Default for unclear but not invalid flows
    
    def _is_subsequence(self, pattern: List[str], sequence: List[str]) -> bool:
        """Check if pattern is a subsequence of sequence"""
        if not pattern:
            return True
        
        pattern_idx = 0
        for item in sequence:
            if pattern_idx < len(pattern) and item == pattern[pattern_idx]:
                pattern_idx += 1
        
        return pattern_idx == len(pattern)

    def _get_action_category(self, action: str) -> str:
        """Get category for an action"""
        for category, actions in self.action_categories.items():
            if action in actions:
                return category
        return 'other'

    def _calculate_semantic_output_match(self, expected_outputs: List[str], 
                                       actual_outputs: List[str]) -> float:
        """Calculate semantic output matching using similarity"""
        if not expected_outputs:
            return 1.0
        
        if not actual_outputs:
            return 0.0
        
        # Use multiple similarity measures
        similarities = []
        for expected in expected_outputs:
            best_similarity = 0.0
            for actual in actual_outputs:
                similarity = self._text_similarity(expected, actual)
                best_similarity = max(best_similarity, similarity)
            similarities.append(best_similarity)
        
        return np.mean(similarities) if similarities else 0.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Text similarity with semantic considerations"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to strings and normalize
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        # Token-based similarity (Jaccard)
        tokens1 = set(re.findall(r'\w+', text1))
        tokens2 = set(re.findall(r'\w+', text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        jaccard_similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        
        # Length-based similarity (penalize very different lengths)
        len1, len2 = len(text1), len(text2)
        length_similarity = 1.0 - abs(len1 - len2) / max(len1, len2)
        
        # Keyword overlap for important terms
        important_terms = {'order', 'user', 'product', 'address', 'payment', 'status'}
        important1 = tokens1 & important_terms
        important2 = tokens2 & important_terms
        keyword_similarity = len(important1 & important2) / len(important1 | important2) if (important1 | important2) else 1.0
        
        # Weighted combination
        return 0.5 * jaccard_similarity + 0.3 * length_similarity + 0.2 * keyword_similarity

    def _extract_outputs(self, model_response: ModelResponse) -> List[str]:
        """Extract outputs with context awareness"""
        outputs = []
        
        # Reasoning as primary output
        if model_response.reasoning:
            outputs.append(model_response.reasoning)
        
        # Tool results with context
        for action in model_response.get_successful_actions():
            if action.result:
                # Add context to tool results
                contextual_result = f"{action.name}: {action.result}"
                outputs.append(contextual_result)
        
        # Error information as negative outputs
        for action in model_response.actions:
            if not action.success and action.error:
                outputs.append(f"Error in {action.name}: {action.error}")
        
        return outputs

    def _determine_success(self, task: Task, model_response: ModelResponse, 
                                  metrics: Dict[str, float], task_type: str = 'unknown') -> bool:
        """Success determination with quality thresholds and task type awareness"""
        # Parsing must succeed
        if not model_response.parsing_success:
            return False
        
        # No expected actions case
        if not task.expected_actions:
            return model_response.parsing_success

        # Adjust thresholds based on task type
        if task_type == 'information':
            # Information tasks: more lenient on action matching, focus on execution
            recall_threshold = 0.5
            sequence_threshold = 0.3
        elif task_type == 'action':
            # Action tasks: stricter on action matching
            recall_threshold = 0.6
            sequence_threshold = 0.4
        else:
            # Mixed or unknown: balanced thresholds
            recall_threshold = 0.55
            sequence_threshold = 0.35

        # Recall threshold with equivalence consideration
        if metrics['recall'] < recall_threshold:
            return False
        
        # Tool execution success requirement
        if model_response.actions:
            success_rate = len(model_response.get_successful_actions()) / len(model_response.actions)
            if success_rate < 0.7:  # Reduced from 0.8 for more flexibility
                return False
        
        # Sequence quality consideration
        if metrics['sequence_quality'] < sequence_threshold:
            return False
        
        return True

    def _calculate_timing_metrics(self, model_response: ModelResponse, total_time: float) -> Dict[str, float]:
        """Calculate comprehensive timing metrics"""
        actions = model_response.actions
        
        tool_execution_time = sum(action.execution_time for action in actions)
        model_thinking_time = model_response.execution_time - tool_execution_time
        
        # Efficiency metrics
        if actions:
            avg_tool_time = tool_execution_time / len(actions)
            time_efficiency = max(0, 1 - avg_tool_time / 5.0)  # 5-second threshold
        else:
            avg_tool_time = 0.0
            time_efficiency = 1.0
        
        return {
            'tool_execution_time': tool_execution_time,
            'model_thinking_time': model_thinking_time,
            'avg_tool_time': avg_tool_time,
            'time_efficiency': time_efficiency,
            'total_efficiency': total_time / max(1, len(actions))  # Time per action
        }

    def _generate_analysis(self, task: Task, model_response: ModelResponse,
                                  metrics: Dict[str, float], timing_metrics: Dict[str, float], task_type: str = 'unknown') -> Dict[str, Any]:
        """Generate comprehensive analysis with quality insights"""
        expected_actions = [action.name for action in task.expected_actions]
        actual_actions = [action.name for action in model_response.actions]
        
        # Action analysis
        correct_actions = list(set(expected_actions) & set(actual_actions))
        missing_actions = list(set(expected_actions) - set(actual_actions))
        extra_actions = list(set(actual_actions) - set(expected_actions))
        
        # Semantic action analysis
        semantic_matches = self._find_semantic_matches(expected_actions, actual_actions)
        
        return {
            'task_type': task_type,
            'expected_actions': expected_actions,
            'actual_actions': actual_actions,
            'correct_actions': correct_actions,
            'missing_actions': missing_actions,
            'extra_actions': extra_actions,
            'semantic_matches': semantic_matches,
            'tool_execution_success_rate': (
                len(model_response.get_successful_actions()) / len(model_response.actions)
                if model_response.actions else 1.0
            ),
            'sequence_quality': metrics['sequence_quality'],
            'timing_efficiency': timing_metrics['time_efficiency'],
            'parsing_details': {
                'parsing_success': model_response.parsing_success,
                'parsing_error': model_response.parsing_error
            },
            'task_metadata': task.metadata
        }

    def _find_semantic_matches(self, expected: List[str], actual: List[str]) -> List[Dict[str, str]]:
        """Find semantic matches between expected and actual actions"""
        matches = []
        expected_used = set()
        actual_used = set()
        
        # Find direct matches
        for exp in expected:
            if exp in actual and exp not in expected_used:
                matches.append({'expected': exp, 'actual': exp, 'type': 'exact'})
                expected_used.add(exp)
                actual_used.add(exp)
        
        # Find semantic matches
        for exp in expected:
            if exp in expected_used:
                continue
            for act in actual:
                if act in actual_used:
                    continue
                if self._are_actions_equivalent(exp, act):
                    matches.append({'expected': exp, 'actual': act, 'type': 'semantic'})
                    expected_used.add(exp)
                    actual_used.add(act)
                    break
        
        return matches

    def _are_actions_equivalent(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically equivalent"""
        for group in self.action_equivalence_groups:
            if action1 in group and action2 in group:
                return True
        return False

    def _generate_intelligent_warnings(self, task: Task, model_response: ModelResponse,
                                     metrics: Dict[str, float], task_type: str = 'unknown') -> List[str]:
        """Generate intelligent warnings with context awareness and task type"""
        warnings = []
        
        # Parsing issues
        if not model_response.parsing_success:
            warnings.append(f"Model response parsing failed: {model_response.parsing_error}")
        
        # Tool execution failures
        failed_tools = [action for action in model_response.actions if not action.success]
        if failed_tools:
            warnings.append(f"{len(failed_tools)} tool(s) failed execution")
            # Add specific failure details for first 2 failures
            for action in failed_tools[:2]:
                warnings.append(f"  - {action.name}: {action.error}")
        
        # Task type specific warnings
        if task_type == 'information':
            # For information tasks, focus on whether info was gathered
            if not model_response.actions:
                warnings.append("Information request but no tools called")
        elif task_type == 'action':
            # For action tasks, focus on whether modifications were made
            modification_tools = self.action_categories.get('modification', set())
            actual_mods = set(action.name for action in model_response.actions) & modification_tools
            expected_mods = set(action.name for action in task.expected_actions) & modification_tools
            if expected_mods and not actual_mods:
                warnings.append("Action request but no modification tools called")
        
        # Action coverage warnings (adjusted by task type)
        recall_threshold = 0.7 if task_type != 'information' else 0.5
        if metrics['recall'] < recall_threshold:
            missing_count = len(set(action.name for action in task.expected_actions) - 
                              set(action.name for action in model_response.actions))
            warnings.append(f"Missing {missing_count} expected actions (recall: {metrics['recall']:.1%})")
        
        # Sequence quality warnings (more lenient)
        if metrics['sequence_quality'] < 0.5:
            warnings.append("Action sequence shows suboptimal logical flow")
        
        # Efficiency warnings
        if model_response.actions:
            avg_time = sum(action.execution_time for action in model_response.actions) / len(model_response.actions)
            if avg_time > 3.0:  # 3-second threshold
                warnings.append(f"Slow tool execution (avg: {avg_time:.1f}s)")
        
        return warnings

    def _calculate_overall_quality_score(self, metrics: Dict[str, float], 
                                       output_match: float, 
                                       timing_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        weights = {
            'action_recall': 0.25,
            'action_precision': 0.20,
            'sequence_quality': 0.20,
            'output_match': 0.15,
            'time_efficiency': 0.10,
            'tool_success': 0.10
        }
        
        # Use F1 as base for action performance
        action_score = metrics['f1']
        
        # Combine all components
        quality_score = (
            weights['action_recall'] * metrics['recall'] +
            weights['action_precision'] * metrics['precision'] +
            weights['sequence_quality'] * metrics['sequence_quality'] +
            weights['output_match'] * output_match +
            weights['time_efficiency'] * timing_metrics['time_efficiency'] +
            weights['tool_success'] * (timing_metrics.get('success_rate', 1.0))
        )
        
        return min(1.0, quality_score)  # Cap at 1.0

    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.4:
            return "Poor"
        else:
            return "Unacceptable"

    def _generate_improvement_suggestions(self, metrics: Dict[str, float], 
                                        timing_metrics: Dict[str, float], task_type: str = 'unknown') -> List[str]:
        """Generate targeted improvement suggestions based on task type"""
        suggestions = []
        
        # Task type specific suggestions
        if task_type == 'information':
            if metrics['recall'] < 0.6:
                suggestions.append("For information requests: ensure all relevant data is retrieved")
        elif task_type == 'action':
            if metrics['recall'] < 0.7:
                suggestions.append("For action requests: ensure all expected modifications are executed")
            if metrics['precision'] < 0.7:
                suggestions.append("Reduce unnecessary actions - focus on requested modifications only")
        else:
            # General suggestions
            if metrics['recall'] < 0.7:
                suggestions.append("Improve action coverage by ensuring all expected actions are executed")
            if metrics['precision'] < 0.7:
                suggestions.append("Reduce unnecessary actions to improve precision")
        
        if metrics['sequence_quality'] < 0.6:
            suggestions.append("Consider alternative logical flows - multiple valid sequences exist")
        
        if timing_metrics['time_efficiency'] < 0.7:
            suggestions.append("Improve tool execution efficiency")
        
        if not suggestions:  # If all metrics are good
            suggestions.append("Maintain current performance levels")
        
        return suggestions


class TaskLoader:
    """Loads tasks from various file formats"""
    
    @staticmethod
    def load_tasks(file_path: str) -> List[Task]:
        """Load tasks from a file generated by task_generator.py or other formats"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tasks = []
            
            if isinstance(data, list):
                # List of tasks
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Handle both direct tasks and wrapped generator results
                        if 'task' in item and isinstance(item['task'], dict):
                            # This is a generator result with metadata
                            task_id = f"task_{i+1:03d}"
                            task = Task.from_dict(item, task_id)
                            tasks.append(task)
                        elif 'q' in item or 'agt' in item:
                            # This is a direct task dict
                            task_id = f"task_{i+1:03d}"
                            task = Task.from_dict(item, task_id)
                            tasks.append(task)
                        else:
                            logger.warning(f"Skipping item at index {i}: no valid task structure")
            
            elif isinstance(data, dict):
                # Single task
                if 'task' in data and isinstance(data['task'], dict):
                    task = Task.from_dict(data, "single_task")
                    tasks.append(task)
                elif 'q' in data or 'agt' in data:
                    task = Task.from_dict(data, "single_task")
                    tasks.append(task)
                else:
                    logger.warning("Single task has no valid task structure")
            
            logger.info(f"Loaded {len(tasks)} valid tasks from {file_path}")
            return tasks
            
        except FileNotFoundError:
            logger.error(f"Task file not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in task file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading tasks from {file_path}: {e}")
            return []


class TaskTester:
    """Main class that orchestrates the testing system with evaluation"""
    
    def __init__(self,
                 envs_path: str = "envs/retail",
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None,
                 user_api_key: str = None,
                 user_base_url: str = None,
                 user_model: str = None,
                 use_dual_model: bool = False,
                 use_mock_model: bool = False,):
        self.envs_path = envs_path
        self.use_dual_model = use_dual_model
        self.use_mock_model = use_mock_model

        # Initialize components
        self.tool_executor = ToolExecutor(envs_path)
        self.task_loader = TaskLoader()
        self.configs = TauBenchConfig()
        

        self.evaluation_engine = EvaluationEngine()
        
        # If mock mode requested, use MockModelClient (fast, deterministic)
        if use_mock_model:
            self.model_client = MockModelClient(self.tool_executor)
            logger.info("Task Tester initialized in mock mode (dry-run)")
        elif use_dual_model:
            # Initialize dual-model client
            self.model_client = DualModelClient(
                user_api_key=user_api_key,
                user_base_url=user_base_url,
                user_model=user_model,
                assistant_api_key=api_key,
                assistant_base_url=base_url,
                assistant_model=model,
                tool_executor=self.tool_executor
            )
            logger.info(f"Task Tester initialized with dual-model approach - User: {user_model or self.configs.user_model}, Assistant: {model or self.configs.default_model}")
        else:
            # Use legacy single model approach for backward compatibility
            self.model_client = ModelClient(api_key, base_url, model, self.tool_executor)
            logger.info(f"Task Tester initialized with single model: {model or self.configs.default_model}")

    def validate_generated_tasks(self, tasks: List[Task]) -> List[ValidationReport]:
        """Validate generated tasks using TaskValidator (no model calls, fast).

        Returns a list of ValidationReport objects (from task_generator.TaskValidator).
        """
        reports: List[ValidationReport] = []
        try:
            # Use the same data_reader underlying the tool executor to validate tasks
            data_reader = self.tool_executor.data_reader if hasattr(self.tool_executor, 'data_reader') else TauBenchDataReader(str(self.envs_path))
            validator = TaskValidator(data_reader)

            for task in tasks:
                # Convert Task back to generator dict format expected by validator
                task_dict = {
                    'q': task.query,
                    'agt': [a.to_dict() for a in task.expected_actions],
                    'ogt': task.expected_outputs,
                    'metadata': task.metadata
                }
                report = validator.validate(task_dict)
                reports.append(report)

            return reports
        except Exception as e:
            logger.error(f"Validation run failed: {e}")
            return reports
    
    def test_single_task(self, task: Task, verbose: bool = False, enhance_query: bool = False) -> TestResult:
        """Test a single task with optional query enhancement and evaluation"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧪 Testing Task: {task.task_id}")
            print(f"Query: {task.query[:100]}{'...' if len(task.query) > 100 else ''}")
            print(f"Expected Actions: {len(task.expected_actions)}")
            print(f"📊 Using Evaluation Engine")
        
        start_time = time.time()
        
        try:
            # Enhance task query if using dual-model and enhancement is requested
            test_task = task
            if self.use_dual_model and enhance_query and hasattr(self.model_client, 'enhance_task_with_user_model'):
                test_task = self.model_client.enhance_task_with_user_model(task)
                if verbose and test_task.query != task.query:
                    print(f"Query: {test_task.query[:100]}{'...' if len(test_task.query) > 100 else ''}")
            
            # Get available tools
            available_tools = self.tool_executor.get_available_tools()
            
            # Get model response
            model_response = self.model_client.get_model_response(test_task, available_tools)
            
            # Execute tools if model response was successful
            if model_response.parsing_success and model_response.actions:
                # In dry-run / mock mode we DO NOT execute real tools; mark actions as mock-executed
                if self.use_mock_model:
                    for action in model_response.actions:
                        action.success = True
                        action.error = None
                        if not action.result:
                            action.result = "[mock executed]"
                        action.execution_time = 0.0
                    # No execution, actions are as-is (treated as executed)
                else:
                    # Partition actions into executable (registered tools) and non-executable (hallucinated)
                    available_names = set(self.tool_executor._tools.keys()) if hasattr(self.tool_executor, '_tools') else set()
                    executable_actions = []
                    non_executable = []

                    for action in model_response.actions:
                        try:
                            raw_name = action.name or ''
                            # sanitize similarly to ToolExecutor.execute_tool
                            sanitized = re.sub(r"<\|.*?\|>.*$", "", raw_name)
                            sanitized = sanitized.split(":")[0].split(" ")[0]
                            lookup_name = sanitized.strip().strip('\"\'')
                        except Exception:
                            lookup_name = (action.name or '').strip()

                        if lookup_name in available_names:
                            # canonicalize name for execution
                            action.name = lookup_name
                            executable_actions.append(action)
                        else:
                            # Treat non-registered tools as non-executable: convert to output and mark as not executed
                            action.success = False
                            action.error = f"Tool '{action.name}' not available; converted to output"
                            # Try to extract user-facing message from arguments
                            msg = None
                            if isinstance(action.arguments, dict):
                                msg = action.arguments.get('message') or action.arguments.get('content') or action.arguments.get('text')
                            if not msg and isinstance(action.arguments, str):
                                msg = action.arguments
                            if msg:
                                # Append to model reasoning so outputs capture the communication
                                model_response.reasoning = (model_response.reasoning or '') + "\n[converted_output] " + str(msg)
                            non_executable.append(action)

                    # Execute only the registered tools
                    executed_actions = self.tool_executor.execute_tools(executable_actions) if executable_actions else []

                    # Reconstruct model_response.actions: executed first, then non-executable placeholders
                    model_response.actions = executed_actions + non_executable
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Evaluate the result using engine
            result = self.evaluation_engine.evaluate(task, model_response, total_time)
            
            if verbose:
                self._display_result(result)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Task execution failed for {task.task_id}: {e}")
            
            return TestResult(
                task_id=task.task_id,
                success=False,
                task=task,
                model_response=ModelResponse(actions=[], parsing_success=False, parsing_error=str(e)),
                total_execution_time=total_time,
                error=str(e)
            )
    
    def test_multiple_tasks(self, 
                           tasks: List[Task], 
                           max_workers: int = 1,
                           verbose: bool = False,
                           enhance_query: bool = False,
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[TestResult]:
        """Test multiple tasks with optional parallel processing and evaluation"""
        
        if max_workers <= 1:
            # Sequential processing
            results = []
            for i, task in enumerate(tasks):
                result = self.test_single_task(task, verbose, enhance_query)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(tasks))
                elif verbose:
                    quality_info = ""
                    if hasattr(result, 'details') and 'quality_rating' in result.details:
                        quality_info = f" | Quality: {result.details['quality_rating']}"
                    print(f"✅ Task {i+1}/{len(tasks)} completed - Success: {result.success}{quality_info}")
            
            return results
        
        else:
            # Parallel processing
            results = [None] * len(tasks)  # Pre-allocate to maintain order
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(self.test_single_task, task, False, enhance_query): i
                    for i, task in enumerate(tasks)
                }
                
                completed = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        logger.error(f"Task {index} failed: {e}")
                        results[index] = TestResult(
                            task_id=f"task_{index:03d}",
                            success=False,
                            task=tasks[index],
                            model_response=ModelResponse(actions=[], parsing_success=False, parsing_error=str(e)),
                            error=str(e)
                        )
                    
                    if progress_callback:
                        progress_callback(completed, len(tasks))
                    elif verbose:
                        quality_info = ""
                        result_obj = results[index]
                        if (hasattr(result_obj, 'details') and result_obj.details and 
                            'quality_rating' in result_obj.details):
                            quality_info = f" | Quality: {result_obj.details['quality_rating']}"
                        print(f"✅ Task {completed}/{len(tasks)} completed - Success: {result_obj.success}{quality_info}")
            
            return results
    
    def _display_result(self, result: TestResult):
        """Display results with quality metrics"""
        print(f"\n📊 Results for {result.task_id}:")
        print(f"  Success: {'✅' if result.success else '❌'}")
        
        # Display quality rating if available
        if hasattr(result, 'details') and 'quality_rating' in result.details:
            quality_rating = result.details['quality_rating']
            quality_score = result.details.get('quality_score', 0)
            print(f"  Quality Rating: {quality_rating} ({quality_score:.1%})")
        
        print(f"  Total Time: {result.total_execution_time:.2f}s")
        print(f"  Model Response Time: {result.model_response_time:.2f}s")
        print(f"  Tool Execution Time: {result.tool_execution_time:.2f}s")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        # Display metrics
        print(f"\n  Core Performance Metrics:")
        print(f"    Action Precision: {result.action_precision:.2%}")
        print(f"    Action Recall: {result.action_recall:.2%}")
        print(f"    Action F1 Score: {result.action_f1:.2%}")
        print(f"    Exact Action Match: {'✅' if result.exact_action_match else '❌'}")
        print(f"    Output Match Rate: {result.output_match_rate:.2%}")
        
        # Display metrics if available
        if hasattr(result, 'details') and 'sequence_quality' in result.details:
            print(f"\n  Quality Metrics:")
            print(f"    Sequence Quality: {result.details['sequence_quality']:.2%}")
            
            efficiency_metrics = result.details.get('efficiency_metrics', {})
            if efficiency_metrics:
                print(f"    Time Efficiency: {efficiency_metrics.get('time_efficiency', 0):.2%}")
                print(f"    Avg Tool Time: {efficiency_metrics.get('avg_tool_time', 0):.2f}s")
        
        if result.warnings:
            print(f"\n  ⚠️  Warnings:")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                print(f"    - {warning}")
        
        # Display improvement suggestions
        if hasattr(result, 'details') and 'improvement_suggestions' in result.details:
            suggestions = result.details['improvement_suggestions']
            if suggestions:
                print(f"\n  💡 Improvement Suggestions:")
                for suggestion in suggestions[:2]:  # Show first 2 suggestions
                    print(f"    - {suggestion}")
        
        # Display action details
        details = result.details
        if details and 'expected_actions' in details:
            print(f"\n  Action Analysis:")
            print(f"    Expected: {details.get('expected_actions', [])}")
            print(f"    Actual: {details.get('actual_actions', [])}")
            
            if details.get('missing_actions'):
                print(f"    Missing: {details['missing_actions']}")
            if details.get('extra_actions'):
                print(f"    Extra: {details['extra_actions']}")
            
            # Display semantic matches if available
            if details.get('semantic_matches'):
                print(f"    Semantic Matches: {len(details['semantic_matches'])}")
        
        # Display reasoning if available
        if result.model_response.reasoning:
            print(f"\n💭 Model Reasoning:")
            reasoning_preview = result.model_response.reasoning[:200]
            if len(result.model_response.reasoning) > 200:
                reasoning_preview += "..."
            print(f"  {reasoning_preview}")
        
        # Display successful tool executions
        successful_tools = result.model_response.get_successful_actions()
        if successful_tools:
            print(f"\n🔧 Tool Executions ({len(successful_tools)} successful):")
            for tool in successful_tools[:3]:  # Show first 3
                result_preview = tool.result[:100] if tool.result else 'No result'
                if tool.result and len(tool.result) > 100:
                    result_preview += "..."
                print(f"  - {tool.name} ({tool.execution_time:.3f}s): {result_preview}")
    
    def generate_summary_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive summary report with metrics"""
        if not results:
            return {"error": "No test results available"}
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        
        # Calculate average metrics
        avg_precision = sum(r.action_precision for r in results) / total_tasks
        avg_recall = sum(r.action_recall for r in results) / total_tasks
        avg_f1 = sum(r.action_f1 for r in results) / total_tasks
        avg_output_match = sum(r.output_match_rate for r in results) / total_tasks
        avg_total_time = sum(r.total_execution_time for r in results) / total_tasks
        avg_model_time = sum(r.model_response_time for r in results) / total_tasks
        avg_tool_time = sum(r.tool_execution_time for r in results) / total_tasks
        
        exact_matches = sum(1 for r in results if r.exact_action_match)
        
        # Calculate metrics if available
        quality_scores = []
        sequence_qualities = []
        efficiency_scores = []
        
        for result in results:
            if hasattr(result, 'details') and result.details:
                if 'quality_score' in result.details:
                    quality_scores.append(result.details['quality_score'])
                if 'sequence_quality' in result.details:
                    sequence_qualities.append(result.details['sequence_quality'])
                if 'efficiency_metrics' in result.details:
                    efficiency_metrics = result.details['efficiency_metrics']
                    if 'time_efficiency' in efficiency_metrics:
                        efficiency_scores.append(efficiency_metrics['time_efficiency'])
        
        # Quality rating distribution
        quality_ratings = []
        for result in results:
            if hasattr(result, 'details') and 'quality_rating' in result.details:
                quality_ratings.append(result.details['quality_rating'])
        
        # Calculate percentiles for timing
        total_times = sorted([r.total_execution_time for r in results])
        model_times = sorted([r.model_response_time for r in results])
        
        def percentile(data, p):
            return data[int(len(data) * p / 100)] if data else 0
        
        # Collect warnings and errors
        all_warnings = []
        all_errors = []
        for result in results:
            all_warnings.extend(result.warnings)
            if result.error:
                all_errors.append(result.error)
        
        # Analysis
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / total_tasks,
                "exact_action_matches": exact_matches,
                "exact_match_rate": exact_matches / total_tasks,
                "total_warnings": len(all_warnings),
                "total_errors": len(all_errors)
            },
            "core_metrics": {
                "average_action_precision": avg_precision,
                "average_action_recall": avg_recall,
                "average_action_f1": avg_f1,
                "average_output_match_rate": avg_output_match
            },
            "metrics": {
                "average_quality_score": np.mean(quality_scores) if quality_scores else 0,
                "average_sequence_quality": np.mean(sequence_qualities) if sequence_qualities else 0,
                "average_efficiency_score": np.mean(efficiency_scores) if efficiency_scores else 0,
                "quality_rating_distribution": dict(Counter(quality_ratings)) if quality_ratings else {}
            },
            "timing": {
                "average_total_time": avg_total_time,
                "average_model_time": avg_model_time,
                "average_tool_time": avg_tool_time,
                "total_time_percentiles": {
                    "p50": percentile(total_times, 50),
                    "p90": percentile(total_times, 90),
                    "p99": percentile(total_times, 99)
                },
                "model_time_percentiles": {
                    "p50": percentile(model_times, 50),
                    "p90": percentile(model_times, 90),
                    "p99": percentile(model_times, 99)
                }
            },
            "error_analysis": {
                "unique_warnings": list(set(all_warnings)),
                "unique_errors": list(set(all_errors)),
                "most_common_warnings": self._count_frequencies(all_warnings)[:5],
                "most_common_errors": self._count_frequencies(all_errors)[:5]
            }
        }
        
        return report
    
    def _count_frequencies(self, items: List[str]) -> List[Tuple[str, int]]:
        """Count frequency of items and return sorted by frequency"""
        from collections import Counter
        return Counter(items).most_common()
    
    def save_results(self, results: List[TestResult], output_dir: str = "results") -> str:
        """Save results to files with comprehensive metadata"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
        
        # Convert results to serializable format with metadata
        serializable_results = []
        for result in results:
            result_dict = {
                "task_id": result.task_id,
                "success": result.success,
                "action_precision": result.action_precision,
                "action_recall": result.action_recall,
                "action_f1": result.action_f1,
                "exact_action_match": result.exact_action_match,
                "output_match_rate": result.output_match_rate,
                "total_execution_time": result.total_execution_time,
                "model_response_time": result.model_response_time,
                "tool_execution_time": result.tool_execution_time,
                "error": result.error,
                "warnings": result.warnings,
                "details": result.details
            }
            
            # Add quality metrics
            if hasattr(result, 'details') and result.details:
                result_dict["quality_metrics"] = {
                    "quality_score": result.details.get('quality_score'),
                    "quality_rating": result.details.get('quality_rating'),
                    "sequence_quality": result.details.get('sequence_quality'),
                    "efficiency_metrics": result.details.get('efficiency_metrics', {}),
                    "improvement_suggestions": result.details.get('improvement_suggestions', [])
                }
            
            # Add task info
            result_dict["task"] = {
                "query": result.task.query,
                "expected_actions": [action.to_dict() for action in result.task.expected_actions],
                "expected_outputs": result.task.expected_outputs,
                "metadata": result.task.metadata
            }
            
            # Add model response info
            result_dict["model_response"] = {
                "actions": [action.to_dict() for action in result.model_response.actions],
                "reasoning": result.model_response.reasoning,
                "parsing_success": result.model_response.parsing_success,
                "parsing_error": result.model_response.parsing_error,
                "execution_time": result.model_response.execution_time
            }
            
            serializable_results.append(result_dict)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        report = self.generate_summary_report(results)
        report_file = os.path.join(output_dir, f"test_summary_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, f"test_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # CSV headers
            writer.writerow([
                "Task ID", "Success", "Quality Rating", "Quality Score",
                "Action Precision", "Action Recall", "Action F1",
                "Sequence Quality", "Output Match", "Total Time", 
                "Model Time", "Tool Time", "Efficiency Score",
                "Error", "Warnings Count"
            ])
            
            for result in results:
                quality_score = result.details.get('quality_score', 0) if hasattr(result, 'details') and result.details else 0
                quality_rating = result.details.get('quality_rating', 'N/A') if hasattr(result, 'details') and result.details else 'N/A'
                sequence_quality = result.details.get('sequence_quality', 0) if hasattr(result, 'details') and result.details else 0
                efficiency_score = result.details.get('efficiency_metrics', {}).get('time_efficiency', 0) if hasattr(result, 'details') and result.details else 0
                
                writer.writerow([
                    result.task_id,
                    result.success,
                    quality_rating,
                    quality_score,
                    result.action_precision,
                    result.action_recall,
                    result.action_f1,
                    sequence_quality,
                    result.output_match_rate,
                    result.total_execution_time,
                    result.model_response_time,
                    result.tool_execution_time,
                    efficiency_score,
                    result.error or "",
                    len(result.warnings)
                ])
        
        logger.info(f"Results saved to {results_file}, {report_file}, and {csv_file}")
        return results_file

    def visualize_results(self, results: List[TestResult], output_dir: str = "results"):
        """Create visualizations of test results"""

        if not results:
            logger.warning("No results to visualize")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        fig.suptitle('Task Tester Results - Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Success rate and quality distribution
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count
        
        # Quality ratings distribution
        quality_ratings = []
        for result in results:
            if hasattr(result, 'details') and 'quality_rating' in result.details:
                quality_ratings.append(result.details['quality_rating'])
        
        if quality_ratings:
            rating_counts = Counter(quality_ratings)
            axes[0, 0].pie(rating_counts.values(), labels=rating_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Quality Rating Distribution')
        else:
            axes[0, 0].pie([success_count, fail_count], 
                          labels=['Success', 'Failure'], 
                          autopct='%1.1f%%',
                          colors=['#4CAF50', '#F44336'])
            axes[0, 0].set_title('Task Success Rate')
        
        # 2. Metrics box plot
        metrics_data = {}
        
        quality_scores = [r.details.get('quality_score', 0) for r in results if hasattr(r, 'details') and r.details]
        sequence_qualities = [r.details.get('sequence_quality', 0) for r in results if hasattr(r, 'details') and r.details]
        
        if quality_scores:
            metrics_data['Quality Score'] = quality_scores
        if sequence_qualities:
            metrics_data['Sequence Quality'] = sequence_qualities
        
        if metrics_data:
            bp = axes[0, 1].boxplot(metrics_data.values(), labels=metrics_data.keys())
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].set_title(' Quality Metrics')
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            # Fallback to basic metrics
            metrics_data = {
                'Precision': [r.action_precision for r in results],
                'Recall': [r.action_recall for r in results],
                'F1': [r.action_f1 for r in results]
            }
            bp = axes[0, 1].boxplot(metrics_data.values(), labels=metrics_data.keys())
            axes[0, 1].set_ylim(0, 1.1)
            axes[0, 1].set_title('Action Performance Metrics')
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Quality vs Efficiency scatter plot
        quality_scores = []
        efficiency_scores = []
        for result in results:
            if hasattr(result, 'details') and result.details:
                quality_score = result.details.get('quality_score')
                efficiency_score = result.details.get('efficiency_metrics', {}).get('time_efficiency')
                if quality_score is not None and efficiency_score is not None:
                    quality_scores.append(quality_score)
                    efficiency_scores.append(efficiency_score)
        
        if quality_scores and efficiency_scores:
            axes[1, 0].scatter(quality_scores, efficiency_scores, alpha=0.6, c='purple')
            axes[1, 0].set_xlabel('Quality Score')
            axes[1, 0].set_ylabel('Time Efficiency')
            axes[1, 0].set_title('Quality vs Efficiency')
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        else:
            # Fallback: Output match rate histogram
            output_matches = [r.output_match_rate for r in results]
            axes[1, 0].hist(output_matches, bins=10, color='#2196F3', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Output Match Rate Distribution')
            axes[1, 0].set_xlabel('Match Rate')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Execution time breakdown with efficiency
        model_times = [r.model_response_time for r in results]
        tool_times = [r.tool_execution_time for r in results]
        
        axes[1, 1].scatter(model_times, tool_times, alpha=0.6)
        axes[1, 1].set_xlabel('Model Response Time (s)')
        axes[1, 1].set_ylabel('Tool Execution Time (s)')
        axes[1, 1].set_title('Model vs Tool Execution Times')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 5. Quality score by execution time
        total_times = [r.total_execution_time for r in results]
        quality_scores = [r.details.get('quality_score', 0.5) for r in results if hasattr(r, 'details') and r.details]
        
        if len(quality_scores) == len(total_times):
            scatter = axes[2, 0].scatter(total_times, quality_scores, c=quality_scores, cmap='viridis', alpha=0.7)
            axes[2, 0].set_xlabel('Total Execution Time (s)')
            axes[2, 0].set_ylabel('Quality Score')
            axes[2, 0].set_title('Quality Score vs Execution Time')
            axes[2, 0].grid(True, linestyle='--', alpha=0.7)
            plt.colorbar(scatter, ax=axes[2, 0])
        else:
            # Fallback: Success rate by execution time
            success_values = [1 if r.success else 0 for r in results]
            colors = ['#4CAF50' if s else '#F44336' for s in success_values]
            axes[2, 0].scatter(total_times, success_values, c=colors, alpha=0.7)
            axes[2, 0].set_xlabel('Total Execution Time (s)')
            axes[2, 0].set_ylabel('Success (1) / Failure (0)')
            axes[2, 0].set_title('Success Rate vs Execution Time')
            axes[2, 0].grid(True, linestyle='--', alpha=0.7)
            axes[2, 0].set_ylim(-0.1, 1.1)
        
        # 6. Sequence quality and warnings
        sequence_qualities = [r.details.get('sequence_quality', 0.5) for r in results if hasattr(r, 'details') and r.details]
        warning_counts = [len(r.warnings) for r in results]
        
        if sequence_qualities and len(sequence_qualities) == len(warning_counts):
            axes[2, 1].scatter(warning_counts, sequence_qualities, alpha=0.6, color='#FF9800')
            axes[2, 1].set_xlabel('Number of Warnings')
            axes[2, 1].set_ylabel('Sequence Quality')
            axes[2, 1].set_title('Sequence Quality vs Warnings')
            axes[2, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            # Fallback: Exact match rate and warnings
            exact_matches = [1 if r.exact_action_match else 0 for r in results]
            axes[2, 1].scatter(warning_counts, exact_matches, alpha=0.6, color='#FF9800')
            axes[2, 1].set_xlabel('Number of Warnings')
            axes[2, 1].set_ylabel('Exact Action Match (1) / No Match (0)')
            axes[2, 1].set_title('Exact Matches vs Warnings')
            axes[2, 1].grid(True, linestyle='--', alpha=0.7)
            axes[2, 1].set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = os.path.join(output_dir, f"Test_visualization_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {viz_file}")




def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description=" Task Tester for GPT OSS 120B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test tasks from generator output (single model)
  python task_tester.py --tasks generated_tasks.json

  # Test with dual-model approach (GPT-4o as user, GPT-OSS-120b as assistant)
  python task_tester.py --tasks tasks.json --dual-model --enhance-query

  # Test with parallel processing and dual models
  python task_tester.py --tasks tasks.json --dual-model --threads 4

  # Test sample task with verbose output and visualization
  python task_tester.py --sample --dual-model --verbose --visualize
        """
    )
    
    # Input options
    parser.add_argument("--tasks", type=str, help="JSON file containing tasks to test")
    
    #Dual-model options
    parser.add_argument("--dual-model", action="store_true", help="Use dual-model approach (user + assistant models)")
    parser.add_argument("--enhance-query", action="store_true", help="Enhance task query using user model")
    
    # Assistant model configuration (GPT-OSS-120b)
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, help="OpenAI API base URL")

    # User model configuration (GPT-4o)
    parser.add_argument("--user-model", type=str, help="User model for dual-model approach")
    parser.add_argument("--user-api-key", type=str, help="User model API key")
    parser.add_argument("--user-base-url", type=str, help="User model API base URL")
    
    # Tool configuration
    parser.add_argument("--envs-path", type=str, default="envs/retail", 
                       help="Path to retail environment")
    
    # Execution options
    parser.add_argument("--threads", type=int, default=1, 
                       help="Number of worker threads for parallel processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Use mock model (dry-run, no LLM calls)")
    parser.add_argument("--validate-only", action="store_true", help="Validate generated tasks using TaskValidator only (no model calls)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Directory to save results")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save detailed results to files")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate result visualizations")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.enhance_query and not args.dual_model:
        print("❌ --enhance-query requires --dual-model")
        return 1

    print("Task Tester for GPT OSS 120B with Dual-Model Support")
    print("=" * 60)
    
    # Initialize tester
    try:
        if args.dual_model:
            tester = TaskTester(
                envs_path=args.envs_path,
                use_dual_model=True,
                use_mock_model=args.dry_run,
                user_api_key=args.user_api_key,
                user_base_url=args.user_base_url,
                user_model=args.user_model,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model
            )
            print(f"✅ Initialized with dual-model approach")
            print(f"   User Model: {args.user_model}")
            print(f"   Assistant Model: {args.model}")
        else:
            tester = TaskTester(
                envs_path=args.envs_path,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                use_mock_model=args.dry_run
            )
            print(f"✅ Initialized with model: {args.model}")
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        print("Make sure the 'envs/retail' directory exists with proper tool implementations")
        return 1
    
    # Load tasks
    tasks = []
    
    if args.tasks:
        tasks = tester.task_loader.load_tasks(args.tasks)
        if not tasks:
            print(f"❌ Failed to load tasks from {args.tasks}")
            return 1
        print(f"📁 Loaded {len(tasks)} tasks from {args.tasks}")
    
    else:
        print("❌ Please specify --tasks")
        return 1
    
    # Validation-only mode (fast): validate generated tasks without model calls
    if args.validate_only:
        print(f"\n🔎 Validating {len(tasks)} generated tasks using TaskValidator (no model calls)...")
        reports = tester.validate_generated_tasks(tasks)
        total = len(reports)
        valid = sum(1 for r in reports if r.valid)
        print(f"Validation: {valid}/{total} valid tasks")
        # Print top sample suggestions/warnings
        for i, rep in enumerate(reports[:3]):
            print(f"\nTask {i+1} - valid={rep.valid}")
            if rep.missing:
                print(f"  Missing: {rep.missing[:3]}")
            if rep.suggestions:
                print(f"  Suggestions: {rep.suggestions[:3]}")
        return 0

    # Run tests
    print(f"\n🧪 Running tests on {len(tasks)} task(s)...")
    if args.dual_model and args.enhance_query:
        print("🔍 Enhancing task queries using user model")

    progress_bar = tqdm(total=len(tasks), desc="Testing tasks")
    def update_progress(current, total):
        try:
            # Calculate the delta between current progress and the bar's current value
            delta = max(0, int(current) - int(progress_bar.n))
            if delta:
                progress_bar.update(delta)
        except Exception:
            # Fallback: refresh the bar to avoid freezing in case of unexpected issues
            try:
                progress_bar.refresh()
            except Exception:
                pass

    start_time = time.time()

    try:
        results = tester.test_multiple_tasks(
            tasks,
            max_workers=args.threads,
            verbose=args.verbose,
            enhance_query=args.enhance_query,
            progress_callback=update_progress
        )
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return 1
    finally:
        if progress_bar:
            progress_bar.close()
    
    total_time = time.time() - start_time
    
    # Generate and display summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY REPORT")
    print(f"{'='*60}")
    
    report = tester.generate_summary_report(results)
    summary = report["summary"]
    core_metrics = report["core_metrics"]
    quality_metrics = report["metrics"]
    timing = report["timing"]
    
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful_tasks']} ({summary['success_rate']:.1%})")
    print(f"Exact Action Matches: {summary['exact_action_matches']} ({summary['exact_match_rate']:.1%})")
    print(f"Total Warnings: {summary['total_warnings']}")
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Total Wall Time: {total_time:.2f}s")
    
    print(f"\nAverage Core Metrics:")
    print(f"  Action Precision: {core_metrics['average_action_precision']:.2%}")
    print(f"  Action Recall: {core_metrics['average_action_recall']:.2%}")
    print(f"  Action F1 Score: {core_metrics['average_action_f1']:.2%}")
    print(f"  Output Match Rate: {core_metrics['average_output_match_rate']:.2%}")
    
    print(f"\nAverage Quality Metrics:")
    print(f"  Quality Score: {quality_metrics['average_quality_score']:.2%}")
    print(f"  Sequence Quality: {quality_metrics['average_sequence_quality']:.2%}")
    print(f"  Efficiency Score: {quality_metrics['average_efficiency_score']:.2%}")
    
    print(f"\nTiming Analysis:")
    print(f"  Average Total Time: {timing['average_total_time']:.2f}s")
    print(f"  Average Model Time: {timing['average_model_time']:.2f}s")
    print(f"  Average Tool Time: {timing['average_tool_time']:.2f}s")
    print(f"  Total Time P50/P90/P99: {timing['total_time_percentiles']['p50']:.2f}s / "
          f"{timing['total_time_percentiles']['p90']:.2f}s / {timing['total_time_percentiles']['p99']:.2f}s")
    
    # Show error analysis if there are issues
    if summary['total_warnings'] > 0 or summary['total_errors'] > 0:
        print(f"\n⚠️  Issue Analysis:")
        error_analysis = report["error_analysis"]
        if error_analysis['most_common_warnings']:
            print(f"  Top Warnings:")
            for warning, count in error_analysis['most_common_warnings']:
                print(f"    - {warning}: {count} times")
        if error_analysis['most_common_errors']:
            print(f"  Top Errors:")
            for error, count in error_analysis['most_common_errors']:
                print(f"    - {error}: {count} times")
    
    # Save results if requested or if there are failures
    if args.save_results or summary['success_rate'] < 1.0:
        print(f"\n💾 Saving results...")
        results_file = tester.save_results(results, args.output_dir)
        print(f"Results saved to: {results_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        print(f"\n📈 Generating visualizations...")
        tester.visualize_results(results, args.output_dir)
    
    print(f"\n🎉 Testing completed!")
    
    # Return appropriate exit code
    return 0 if summary['success_rate'] > 0.8 else 1


if __name__ == "__main__":
    exit(main())