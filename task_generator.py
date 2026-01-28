# task_generator.py
"""
TauBench Task Generator with AgentFlow Architecture

This module provides a complete task generation system for TauBench benchmarks,
following the AgentFlow multi-turn iterative architecture:

AgentFlow Architecture (from research image):
1. **Multi-turn Planning**: Planner analyzes query+memory, outputs actions (a^t)
2. **Action Execution**: Executor runs planned actions, produces results (e^t)
3. **Verification**: Verifier checks execution, outputs verification status (v^t)
4. **Memory Management**: Memory (M^t) tracks state across turns (M^t+1)
5. **Generation**: Generator synthesizes final task from verified executions

Generation Modes:
1. **Direct Mode**: Single-shot generation (legacy, faster)
2. **AgentFlow Mode**: Multi-turn iterative refinement (higher quality)

"""
import json
import copy
import logging
import os
import random
import time
import re
import datetime
import tiktoken
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from openai import OpenAI
from data_reader import TauBenchDataReader
from functools import lru_cache
from configs import TauBenchConfig
from collections import Counter, defaultdict
import weakref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Module-level helpers to avoid duplicated logic across classes
# These centralize order ID normalization and required-field checks.
ORDER_NORMALIZE_RE = re.compile(r'#?W?(\d+)', re.IGNORECASE)


@lru_cache(maxsize=512)
def _normalize_order_id_global(order_id: str) -> str:
    """Normalize order ID into canonical form, e.g. '#W123456'.

    Cached for performance and used across validator/generator logic.
    """
    if not order_id:
        return ''
    s = str(order_id).strip().strip('\"\'')
    m = ORDER_NORMALIZE_RE.match(s)
    if m:
        return f"#W{m.group(1)}"
    return s


def _find_user_by_name_zip_global(first_name: str, last_name: str, zip_code: str, users: Dict) -> Optional[str]:
    """Find a single user id by first/last name and zip code."""
    if not all([first_name, last_name, zip_code]):
        return None
    fn_lower = first_name.strip().lower()
    ln_lower = last_name.strip().lower()
    zip_str = str(zip_code).strip()

    for uid, user in users.items():
        if not isinstance(user, dict):
            continue
        addr = user.get('address', {})
        if isinstance(addr, dict):
            user_zip = str(addr.get('zip') or addr.get('zip_code', '')).strip()
            if user_zip != zip_str:
                continue
        else:
            continue

        # Compare names
        name_val = user.get('name', '')
        if isinstance(name_val, dict):
            name_str = f"{name_val.get('first_name','').strip()} {name_val.get('last_name','').strip()}".strip()
        else:
            name_str = str(name_val or '')

        name_lower = name_str.lower()
        if fn_lower in name_lower and ln_lower in name_lower:
            return uid
    return None


# Response extraction helpers: unify API response extraction and text->JSON fallback parsing


def _extract_json_candidate_from_text(content: str) -> Optional[Any]:
    """Try to find and parse a JSON object anywhere inside arbitrary text.

    This centralizes the fallback heuristics used by ResponseParser and
    makes them available to other callers that need robust JSON extraction.
    """
    if not content or not isinstance(content, str):
        return None

    # Remove common markdown/code fences
    try:
        cleaned = re.sub(r"```(?:json)?\n?|```", "", content, flags=re.I | re.S)
    except Exception:
        cleaned = content

    # Find the first '{' and the last '}' and attempt to parse that slice.
    try:
        first = cleaned.find('{')
        last = cleaned.rfind('}')
        if first != -1 and last != -1 and last > first:
            candidate = cleaned[first:last+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try a slightly relaxed cleanup: remove trailing commas
                candidate2 = re.sub(r",\s*}\s*$", "}", candidate)
                candidate2 = re.sub(r",\s*\]", "]", candidate2)
                try:
                    return json.loads(candidate2)
                except json.JSONDecodeError:
                    logger.debug("Fallback JSON candidate parse failed")
    except Exception as e:
        logger.debug(f"Error during fallback JSON extraction: {e}")

    return None


def _generate_llm_ogt(memory: 'TaskGenerationMemory', api_client: Optional['APIClient'] = None) -> Optional[str]:
    """Generate natural agent response via LLM using provided API client.

    Returns generated text or None on failure.
    """
    try:
        query = memory.query or ''
        actions = memory.actions or []
        user_ctx = memory.prompt_data.get('sampled_user_details', {}) if memory.prompt_data else {}

        prompt = f"""You are a helpful customer service agent. Based on the user's query and the actions taken, generate a natural, professional response.

User Query:
{query}

Actions Taken:
{json.dumps(actions, indent=2)}

User Context:
{json.dumps(user_ctx, indent=2)}

Output only the agent's response text, no JSON or extra formatting."""

        if api_client is None:
            # Attempt to create a local client if available
            return None

        text = api_client.call_with_retry(prompt)
        return text.strip() if text else None
    except Exception as e:
        logger.warning(f"LLM ogt generation failed: {e}")
        return None


# Tool schema cache per-data-reader to avoid duplicate reads/parsing
_TOOL_SCHEMA_CACHE: 'weakref.WeakKeyDictionary' = weakref.WeakKeyDictionary()

def _get_tool_schemas(data_reader: TauBenchDataReader) -> Dict[str, Dict[str, Any]]:
    """Return parsed tool schemas for a data_reader, caching per-instance.

    This avoids repeated calls to read_tools_info and keeps a single parsed
    schema map available to executors and other components.
    """
    try:
        if data_reader in _TOOL_SCHEMA_CACHE:
            return _TOOL_SCHEMA_CACHE[data_reader]
    except Exception:
        # WeakKeyDictionary may raise if unhashable; fall back to no-cache
        pass

    schemas: Dict[str, Dict[str, Any]] = {}
    try:
        tools_info = data_reader.read_tools_info()
        for tool in tools_info:
            if isinstance(tool, dict) and 'function' in tool:
                func = tool['function']
                name = func.get('name')
                if name:
                    schemas[name] = {
                        'parameters': func.get('parameters', {}),
                        'required': func.get('parameters', {}).get('required', [])
                    }
    except Exception as e:
        logger.warning(f"Failed to load tool schemas: {e}")
        schemas = {}

    try:
        _TOOL_SCHEMA_CACHE[data_reader] = schemas
    except Exception:
        # If caching fails, return parsed value anyway
        pass

    return schemas


@dataclass
class ParseResult:
    """Result of parsing an LLM response"""
    success: bool
    task: Optional[Dict[str, Any]] = None
    thought: str = ""
    error: Optional[str] = None
    raw_response: str = ""
    candidates_tried: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Report from validating a generated task"""
    valid: bool
    missing: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)

class ResponseParser:
    """
    Handles parsing of LLM responses with STRICT format enforcement per TauBench.
    
    Expected format:
        <thought>reasoning here</thought>
        <answer>{"q": "...", "agt": [...], "ogt": [...]}</answer>
    
    Performance: Uses string.find() for O(n) parsing, validates structure upfront.
    """
    
    # Constants for tag extraction
    THOUGHT_START = '<thought>'
    THOUGHT_END = '</thought>'
    ANSWER_START = '<answer>'
    ANSWER_END = '</answer>'
    REQUIRED_KEYS = {'q', 'agt', 'ogt'}
    
    @classmethod
    def parse(cls, response_content: str) -> ParseResult:
        """
        Parse LLM response enforcing strict <thought>/<answer> format.
        
        Args:
            response_content: Raw LLM response string
            
        Returns:
            ParseResult with success status and extracted data or error
            
        Raises:
            None - all errors are captured in ParseResult
        """
        if not response_content or not isinstance(response_content, str):
            return ParseResult(
                success=False,
                error="Empty or invalid response content",
                raw_response=str(response_content)
            )
        
        # Extract and validate thought section
        thought = cls._extract_section(
            response_content, 
            cls.THOUGHT_START, 
            cls.THOUGHT_END
        )
        if not thought:
            return ParseResult(
                success=False,
                error="Missing or malformed <thought> section",
                raw_response=response_content
            )
        
        # Extract and parse answer JSON
        answer = cls._extract_answer(response_content)
        if answer is None:
            return ParseResult(
                success=False,
                error="No valid <answer> section found or JSON is malformed",
                raw_response=response_content
            )
        
        # Validate answer structure
        if not isinstance(answer, dict):
            return ParseResult(
                success=False,
                error=f"Answer must be a dict, got {type(answer).__name__}",
                raw_response=response_content
            )
        
        missing_keys = cls.REQUIRED_KEYS - answer.keys()
        if missing_keys:
            return ParseResult(
                success=False,
                error=f"Answer missing required keys: {missing_keys}",
                raw_response=response_content
            )
        
        return ParseResult(
            success=True,
            task=answer,
            thought=thought,
            raw_response=response_content
        )
    
    @classmethod
    def _extract_section(cls, content: str, start_tag: str, end_tag: str) -> str:
        """
        Extract content between XML-style tags.
        
        Args:
            content: Source string
            start_tag: Opening tag (e.g., '<thought>')
            end_tag: Closing tag (e.g., '</thought>')
            
        Returns:
            Extracted content or empty string if not found
        """
        try:
            start_idx = content.find(start_tag)
            end_idx = content.find(end_tag)
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                start_pos = start_idx + len(start_tag)
                return content[start_pos:end_idx].strip()
        except Exception as e:
            logger.debug(f"Failed to extract section {start_tag}: {e}")
        
        return ""
    
    @classmethod
    def _extract_answer(cls, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse answer JSON from <answer> tags.
        
        Args:
            content: Source string containing <answer> section
            
        Returns:
            Parsed JSON dict or None if extraction/parsing fails
        """
        # Try strict extraction first
        answer_str = cls._extract_section(content, cls.ANSWER_START, cls.ANSWER_END)
        if answer_str:
            try:
                return json.loads(answer_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse answer JSON from <answer> section: {e}")

        # Delegate fallback JSON extraction to centralized helper
        try:
            parsed = _extract_json_candidate_from_text(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug(f"Fallback json extraction failed: {e}")

        return None

class TaskValidator:
    """
    Validates generated tasks against real data with optimized performance.
    
    Enhanced with:
    - User-order relationship validation
    - Cross-action consistency checking  
    - Business logic validation
    - Optimized indexing and caching
    """
    
    def __init__(self, data_reader: TauBenchDataReader):
        self.data_reader = data_reader
        self.config = TauBenchConfig()
        self._cached_data = None
        self._indexes = None
        # Compile patterns once
        self._patterns = {
            'name': re.compile(r'([A-Z][a-z]+)\s+([A-Z][a-z]+)'),
            'email': re.compile(r'[\w\.-]+@[\w\.-]+'),
            'order_normalize': re.compile(r'#?W?(\d+)', re.IGNORECASE),
            'zip': re.compile(r'\b(\d{5})\b')
        }
    
    def _get_data_and_indexes(self) -> Tuple[Dict, Dict]:
        """Get cached data and pre-built indexes in single call."""
        if self._cached_data is None:
            self._cached_data = self.data_reader.read_data_files()
            self._indexes = self._build_all_indexes(self._cached_data)
        
        return self._cached_data, self._indexes
    
    def _build_all_indexes(self, data: Dict) -> Dict[str, Any]:
        """Build all required indexes in one pass."""
        users = data.get('users', {})
        orders = data.get('orders', {})
        
        return {
            'email_index': {
                user['email'].lower().strip(): uid
                for uid, user in users.items()
                if isinstance(user, dict) and user.get('email')
            },
            'user_orders_index': self._build_user_orders_index(orders)
        }
    
    def _build_user_orders_index(self, orders: Dict) -> Dict[str, Set[str]]:
        """Build user to orders mapping."""
        index = defaultdict(set)
        for order_id, order in orders.items():
            if isinstance(order, dict) and (user_id := order.get('user_id')):
                index[user_id].add(order_id)
        return index

    def validate(self, task: Dict[str, Any]) -> ValidationReport:
        """Validate task references and relationships against real data."""
        report = ValidationReport(valid=True)
        
        try:
            data, indexes = self._get_data_and_indexes()
            # Single-pass validation
            self._validate_task_complete(task, data, indexes, report)
            self._validate_q_action_consistency(task, report)
            # Single-pass validation with relationship checking
            self._validate_task_complete(task, data, indexes, report)
            
            report.valid = len(report.missing) == 0
            return report
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationReport(
                valid=False,
                missing=[{'type': 'exception', 'message': str(e)}]
            )
    
    def _validate_task_complete(
        self, 
        task: Dict[str, Any], 
        data: Dict,
        indexes: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """Complete task validation in optimized single pass."""
        actions = task.get('agt', [])
        if not actions:
            return
        
        users = data.get('users', {})
        orders = data.get('orders', {})
        products = data.get('products', {})
        
        # Track validation state
        validation_state = {
            'current_user_id': None,
            'user_established': False,
            'missing_entries': set(),
            'action_contexts': []
        }
        
        # Single pass through all actions
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
                
            action_context = self._validate_single_action_complete(
                action, idx, users, orders, products, indexes, validation_state
            )
            validation_state['action_contexts'].append(action_context)

        # Business-logic validation: ensure actions are appropriate for order statuses
        try:
            self._validate_business_rules(task, users, orders, indexes, validation_state)
        except Exception:
            # Be permissive; main validation will catch issues
            pass
        
        # Process collected validation results
        self._process_validation_results(validation_state, report, task, users, indexes)
    
    def _validate_single_action_complete(
        self,
        action: Dict[str, Any],
        action_index: int,
        users: Dict,
        orders: Dict,
        products: Dict,
        indexes: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate single action and update validation state."""
        name = action.get('name', '')
        args = action.get('arguments', {}) or {}
        context = {'action_name': name, 'action_index': action_index}
        
        # Update user context
        state['current_user_id'] = self._determine_current_user(name, args, users, state['current_user_id'])
        if state['current_user_id'] and not state['user_established']:
            state['user_established'] = True
        
        # Validate all reference types
        self._validate_user_references_complete(args, name, users, indexes['email_index'], state, context)
        self._validate_order_references_complete(args, name, orders, state['current_user_id'], indexes['user_orders_index'], state, context)
        self._validate_product_references_complete(args, name, products, state, context)
        self._validate_order_item_relationship(args, name, orders, state, context)  # NEW: Validate item references
        self._validate_special_actions_complete(name, args, users, state, context)
        
        return context
    
    def _determine_current_user(
        self, 
        action_name: str, 
        args: Dict[str, Any], 
        users: Dict,
        current_user: Optional[str]
    ) -> Optional[str]:
        """Determine current user based on action type and arguments."""
        user_id = args.get('user_id')
        
        # User identification actions establish context
        if action_name in ['find_user_id_by_email', 'find_user_id_by_name_zip']:
            return user_id if user_id and user_id in users else current_user
        
        # User-targeting actions maintain context
        elif action_name in self.config.user_target_actions:
            return user_id if user_id and user_id in users else current_user
        
        # Return current context for other actions
        return current_user
    
    def _validate_user_references_complete(
        self,
        args: Dict[str, Any],
        action_name: str,
        users: Dict,
        email_index: Dict[str, str],
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate all user-related references."""
        # User ID validation
        if 'user_id' in args and args['user_id'] not in users:
            state['missing_entries'].add((
                'user_id', args['user_id'], action_name, context['action_index']
            ))
        
        # Email validation
        if 'email' in args:
            self._validate_email_reference(args['email'], action_name, email_index, users, state, context)
    
    def _validate_email_reference(
        self,
        email: str,
        action_name: str,
        email_index: Dict[str, str],
        users: Dict,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate email reference and generate suggestions."""
        email_key = email.lower().strip()
        user_id = email_index.get(email_key)
        
        if not user_id:
            # Email not found
            state['missing_entries'].add((
                'email', email, action_name, context['action_index']
            ))
        elif not context.get('arguments', {}).get('user_id'):
            # Email found but user_id missing - store for suggestion
            context.setdefault('suggestions', []).append({
                'type': 'missing_user_id',
                'suggest_user_id': user_id
            })
    
    def _validate_order_references_complete(
        self,
        args: Dict[str, Any],
        action_name: str,
        orders: Dict,
        current_user_id: Optional[str],
        user_orders_index: Dict[str, Set[str]],
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate all order-related references with relationship checking."""
        order_ids = self._extract_order_ids(args)
        
        for order_id in order_ids:
            normalized_oid = self._normalize_order_id(str(order_id))
            
            # Order existence check
            if normalized_oid not in orders:
                state['missing_entries'].add((
                    'order_id', order_id, action_name, context['action_index']
                ))
                continue
            
            # Relationship validation
            self._validate_order_relationship(
                normalized_oid, order_id, action_name, orders, 
                current_user_id, user_orders_index, state, context
            )
    
    def _validate_order_relationship(
        self,
        normalized_oid: str,
        original_oid: str,
        action_name: str,
        orders: Dict,
        current_user_id: Optional[str],
        user_orders_index: Dict[str, Set[str]],
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate order ownership relationship."""
        order_owner = orders[normalized_oid].get('user_id')
        
        if not order_owner:
            return  # Order has no owner, skip relationship check
        
        if current_user_id:
            # Check ownership and accessibility
            if current_user_id != order_owner:
                state['missing_entries'].add((
                    'order_ownership_mismatch',
                    f'Order {original_oid} belongs to user {order_owner}',
                    action_name, context['action_index']
                ))
            elif (current_user_id in user_orders_index and 
                  normalized_oid not in user_orders_index[current_user_id]):
                state['missing_entries'].add((
                    'order_not_accessible',
                    f'Order {original_oid} not accessible for user {current_user_id}',
                    action_name, context['action_index']
                ))
        else:
            # No user context established
            state['missing_entries'].add((
                'user_context_missing',
                f'Order {original_oid} validation requires user context',
                action_name, context['action_index']
            ))
    
    def _validate_order_item_relationship(
        self,
        args: Dict[str, Any],
        action_name: str,
        orders: Dict,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate that item references match actual order items."""
        order_ids = self._extract_order_ids(args)
        
        for order_id in order_ids:
            normalized_oid = self._normalize_order_id(str(order_id))
            
            if normalized_oid not in orders:
                continue  # Already caught by order validation
            
            order = orders[normalized_oid]
            order_items = order.get('items', [])
            
            if not isinstance(order_items, list):
                continue
            
            # Check item_positions
            if 'item_positions' in args:
                positions = args['item_positions']
                if isinstance(positions, list):
                    max_position = len(order_items)
                    invalid_positions = [p for p in positions if p > max_position or p < 1]
                    
                    if invalid_positions:
                        state['missing_entries'].add((
                            'invalid_item_positions',
                            f'Order {order_id} has {max_position} items, but action references positions {invalid_positions}',
                            action_name, context['action_index']
                        ))
            
            # Check item_ids
            if 'item_ids' in args:
                item_ids = args['item_ids']
                if isinstance(item_ids, list):
                    valid_item_ids = {item.get('item_id') for item in order_items if isinstance(item, dict) and item.get('item_id')}
                    invalid_ids = [iid for iid in item_ids if iid not in valid_item_ids]
                    
                    if invalid_ids:
                        state['missing_entries'].add((
                            'invalid_item_ids',
                                                        f'Order {order_id} does not contain items {invalid_ids}',
                            action_name, context['action_index']
                        ))
    
    def _validate_product_references_complete(
        self,
        args: Dict[str, Any],
        action_name: str,
        products: Dict,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate product references."""
        if 'product_id' in args and args['product_id'] not in products:
            state['missing_entries'].add((
                'product_id', args['product_id'], action_name, context['action_index']
            ))

        # For exchange actions, validate replacement item ids correspond to product variants and are available
        if action_name == 'exchange_delivered_order_items' and 'new_item_ids' in args:
            new_ids = args.get('new_item_ids') or []
            if not isinstance(new_ids, list):
                return
            # Build set of available variant ids across product catalog
            available_variants = set()
            for pid, p in products.items():
                variants = p.get('variants', {}) if isinstance(p.get('variants', {}), dict) else {}
                for vid, v in variants.items():
                    if v.get('available'):
                        available_variants.add(str(vid))

            invalid_new_ids = [nid for nid in new_ids if str(nid) not in available_variants]
            if invalid_new_ids:
                state['missing_entries'].add((
                    'invalid_exchange_variants', f'New item_ids not available: {invalid_new_ids}', action_name, context['action_index']
                ))
    
    def _validate_special_actions_complete(
        self,
        action_name: str,
        args: Dict[str, Any],
        users: Dict,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate special action types that require custom logic."""
        if action_name == 'find_user_id_by_name_zip':
            self._validate_name_zip_action(args, users, state, context)

    def _validate_business_rules(
        self,
        task: Dict[str, Any],
        users: Dict,
        orders: Dict,
        indexes: Dict[str, Any],
        state: Dict[str, Any]
    ) -> None:
        """Validate business rules such as order-status vs action compatibility.

        Adds entries to state['missing_entries'] for mismatches so they appear
        in ValidationReport.missing and produce suggestions.
        """
        # Define actions that are only appropriate for pending orders
        pending_only = {
            'modify_pending_order_address',
            'modify_pending_order_payment',
            'modify_pending_order_items',
            'cancel_pending_order'
        }

        delivered_only = {
            'return_delivered_order_items',
            'exchange_delivered_order_items'
        }

        # Iterate through actions and check order ownership/status
        for idx, action in enumerate(task.get('agt', []) or []):
            if not isinstance(action, dict):
                continue
            name = action.get('name', '')
            args = action.get('arguments', {}) or {}
            order_ids = []
            if 'order_id' in args and args.get('order_id'):
                v = args.get('order_id')
                if isinstance(v, list):
                    order_ids.extend(v)
                else:
                    order_ids.append(v)
            if 'order_ids' in args and isinstance(args.get('order_ids'), list):
                order_ids.extend(args.get('order_ids'))

            for raw_oid in order_ids:
                oid = self._normalize_order_id(str(raw_oid))
                order = orders.get(oid)
                if not order:
                    continue
                status = str(order.get('status', '')).lower()

                # If order is delivered/completed, disallow pending-only actions
                if status in ('delivered', 'completed', 'shipped') and name in pending_only:
                    state['missing_entries'].add((
                        'business_rule_mismatch',
                        f"Action '{name}' is not allowed for order {oid} with status '{status}'",
                        name,
                        idx
                    ))

                # If order is not delivered but action is delivered-only
                if status not in ('delivered', 'completed', 'shipped') and name in delivered_only:
                    state['missing_entries'].add((
                        'business_rule_mismatch',
                        f"Action '{name}' expects a delivered order but {oid} has status '{status}'",
                        name,
                        idx
                    ))

    def _validate_q_action_consistency(self, task: Dict[str, Any], report: ValidationReport) -> None:
        """Ensure natural-language questions (q) does not promise outcomes that aren't implemented by agt.

        This method scans `q` text for keywords that imply actions (refunds, return labels,
        replacements, expedited shipping, tracking) and ensures the corresponding tool calls
        exist in `agt`. If a missing action is detected, it appends a suggestion to the
        ValidationReport so callers (or apply_suggestions) can add the required action.
        """
        try:
            q_list = task.get('q') or []
            if not q_list:
                return
            q_text = " ".join(q_list).lower()

            agt = task.get('agt') or []
            action_names = {a.get('name') for a in agt if isinstance(a, dict) and a.get('name')}

            # Mapping of textual cues to required tool actions (allowed alternatives)
            checks = [
                (['refund', 'refunded', 'refunds'], {'return_delivered_order_items'}),
                (['return label', 'prepaid return', 'return‑shipping', 'return shipping', 'return_label'], {'return_delivered_order_items'}),
                (['replacement', 'replacement will be', 'replacement is', 'replacement shipped'], {'exchange_delivered_order_items', 'return_delivered_order_items'}),
                (['expedited', 'expedite', 'expedited shipping'], {'exchange_delivered_order_items'}),
            ]

            # Load data / indexes so we can validate product/order availability for suggested actions
            try:
                data, indexes = self._get_data_and_indexes()
                orders = data.get('orders', {})
                products = data.get('products', {})
            except Exception:
                orders = {}
                products = {}

            for keywords, required_actions in checks:
                if not any(k in q_text for k in keywords):
                    continue

                # Special handling for refunds: choose return vs cancel based on order status
                if keywords[0] in ('refund', 'refunded', 'refunds'):
                    # Try to infer order ids from actions or the query text
                    order_ids = set()
                    for a in task.get('agt', []) or []:
                        if isinstance(a, dict):
                            order_ids.update(self._extract_order_ids(a.get('arguments', {}) or {}))
                    if not order_ids:
                        if m := re.search(r"#W?\d+", q_text):
                            order_ids.add(self._normalize_order_id(m.group(0)))

                    # If we found orders, suggest based on their status
                    if order_ids:
                        for oid in order_ids:
                            oidn = self._normalize_order_id(str(oid))
                            ordobj = orders.get(oidn)
                            if not ordobj:
                                continue
                            status = str(ordobj.get('status', '')).lower()
                            if status in ('delivered', 'completed', 'shipped'):
                                req_actions = {'return_delivered_order_items'}
                            else:
                                req_actions = {'cancel_pending_order'}

                            if not (action_names & req_actions):
                                suggested = {
                                    'type': 'q_action_mismatch',
                                    'missing_actions': list(req_actions),
                                    'reason': f"Q mentions '{keywords[0]}' but no corresponding action was found in agt; order {oid} status: {status}",
                                    'suggest_action': {
                                        'name': next(iter(req_actions)),
                                        'arguments': {'order_id': oidn}
                                    }
                                }
                                report.suggestions.append(suggested)
                                break
                        continue

                # If exchange is requested, only suggest if exchange is possible for the order and product
                if 'exchange_delivered_order_items' in required_actions:
                    exchange_ok = self._is_exchange_possible(task, orders, products, q_text)
                    if not exchange_ok:
                        # If exchange not possible, suggest return or escalate instead
                        alt_actions = {'return_delivered_order_items', 'transfer_to_human_agents'}
                        if not (action_names & alt_actions):
                            suggested = {
                                'type': 'q_action_mismatch',
                                'missing_actions': list(alt_actions),
                                'reason': f"Q mentions '{keywords[0]}' and exchange was requested but no replacement variant is available; suggesting return or escalate",
                                'suggest_action': {
                                    'name': 'return_delivered_order_items',
                                    'arguments': {}
                                }
                            }
                            report.suggestions.append(suggested)
                        continue

                if not (action_names & required_actions):
                    # Suggest adding at least one of the required actions
                    # Prepare a minimal suggested action (arguments can be filled by apply_suggestions)
                    suggested = {
                        'type': 'q_action_mismatch',
                        'missing_actions': list(required_actions),
                        'reason': f"Q mentions '{keywords[0]}' but no corresponding action was found in agt",
                        'suggest_action': {
                            'name': next(iter(required_actions)),
                            'arguments': {}
                        }
                    }
                    report.suggestions.append(suggested)
        except Exception:
            # Be permissive on unexpected errors here; do not raise
            return
    
    def _validate_name_zip_action(
        self,
        args: Dict[str, Any],
        users: Dict,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Validate find_user_id_by_name_zip action."""
        first_name = args.get('first_name', '').strip()
        last_name = args.get('last_name', '').strip()
        zip_code = str(args.get('zip') or args.get('zip_code', '')).strip()

        if not (first_name and last_name and zip_code):
            state['missing_entries'].add((
                'incomplete_name_zip',
                f"Missing fields: first_name='{first_name}', last_name='{last_name}', zip='{zip_code}'",
                context['action_name'], context['action_index']
            ))
            return

        # Check if combination exists
        found_uid = self._find_user_by_name_zip(first_name, last_name, zip_code, users)
        if not found_uid:
            state['missing_entries'].add((
                'name_zip_mismatch',
                f"No user found for: {first_name} {last_name}, zip: {zip_code}",
                context['action_name'], context['action_index']
            ))
    
    def _process_validation_results(
        self,
        state: Dict[str, Any],
        report: ValidationReport,
        task: Dict[str, Any],
        users: Dict,
        indexes: Dict[str, Any]
    ) -> None:
        """Process collected validation results and generate final report."""
        # Add missing entries to report
        for entry_type, value, action, idx in state['missing_entries']:
            report.missing.append({
                'type': entry_type,
                'value': value,
                'action': action,
                'action_index': idx
            })
        
        # Add suggestions from action contexts
        for context in state.get('action_contexts', []):
            for suggestion in context.get('suggestions', []):
                report.suggestions.append({
                    'action': context['action_name'],
                    'action_index': context['action_index'],
                    **suggestion
                })
        
        # Generate query-based suggestions if needed
        if not self._has_valid_user(task, users):
            self._generate_suggestions_from_query(task, users, indexes['email_index'], report)
        
        # Validate cross-action consistency
        self._validate_task_consistency(state['action_contexts'], report)

        # Validate that q does not promise outcomes that are missing in agt
        try:
            self._validate_q_action_consistency(task, report)
        except Exception:
            pass
    
    def _validate_task_consistency(
        self,
        action_contexts: List[Dict[str, Any]],
        report: ValidationReport
    ) -> None:
        """Validate consistency across all actions in the task."""
        if len(action_contexts) <= 1:
            return
        
        # Check for multiple users used
        user_ids = set()
        for context in action_contexts:
            if user_id := context.get('current_user_id'):
                user_ids.add(user_id)
        
        if len(user_ids) > 1:
            report.missing.append({
                'type': 'multiple_users_detected',
                'value': f"Task uses {len(user_ids)} different users: {user_ids}",
                'action': 'cross_action_validation',
                'action_index': -1
            })
    
    def _extract_order_ids(self, args: Dict[str, Any]) -> List[str]:
        """Extract order_id(s) from an action arguments dict in a consistent way."""
        order_ids: List[str] = []
        if not isinstance(args, dict):
            return order_ids
        if 'order_id' in args and args['order_id']:
            order_ids.append(args['order_id'])
        if 'order_ids' in args and isinstance(args['order_ids'], list):
            order_ids.extend([oid for oid in args['order_ids'] if oid])
        return order_ids
    
    @lru_cache(maxsize=256)
    def _normalize_order_id(self, order_id: str) -> str:
        """Normalize order ID with caching. Delegates to module-level implementation."""
        try:
            return _normalize_order_id_global(order_id)
        except Exception:
            return ''
    
    def _has_valid_user(self, task: Dict[str, Any], users: Dict) -> bool:
        """Check if task contains a valid user_id present in users."""
        for action in task.get('agt', []):
            if isinstance(action, dict):
                args = action.get('arguments', {})
                if args.get('user_id') in users:
                    return True
        return False

    def _generate_suggestions_from_query(
        self, 
        task: Dict[str, Any], 
        users: Dict,
        email_index: Dict[str, str],
        report: ValidationReport
    ) -> None:
        """Generate suggestions from query text."""
        q = task.get('q', '')
        if not q:
            return
        
        # Try name pattern
        if name_match := self._patterns['name'].search(q):
            name_fragment = name_match.group(0)
            if matches := self._find_users_by_name_fast(name_fragment, users):
                report.suggestions.append({
                    'found_name_in_q': name_fragment, 
                    'suggest_user_id': matches[0][0]
                })
                return
        
        # Try email pattern
        if email_match := self._patterns['email'].search(q):
            email = email_match.group(0)
            if uid := email_index.get(email.lower().strip()):
                report.suggestions.append({
                    'found_email_in_q': email, 
                    'suggest_user_id': uid
                })

    def _find_users_by_name_fast(self, name_fragment: str, users: Dict, limit: int = 3) -> List[Tuple[str, Dict]]:
        """Find users by name fragment (case-insensitive).

        Returns up to `limit` matches as (user_id, user_dict).
        """
        if not name_fragment:
            return []
        fragment_lower = name_fragment.lower()
        matches: List[Tuple[str, Dict]] = []
        for uid, user in users.items():
            if not isinstance(user, dict):
                continue
            # Build comparable name
            name_val = user.get('name', '')
            if isinstance(name_val, dict):
                first = name_val.get('first_name', '').strip()
                last = name_val.get('last_name', '').strip()
                name_str = f"{first} {last}".strip()
            else:
                name_str = str(name_val or '')

            if fragment_lower in name_str.lower():
                matches.append((uid, user))
                if len(matches) >= limit:
                    break
        return matches

    @staticmethod
    def _get_user_name_string(name: Any) -> str:
        """Extract name string from dict or string."""
        if isinstance(name, dict):
            first = name.get('first_name', '').strip()
            last = name.get('last_name', '').strip()
        else:
            parts = str(name).split() if name else []
            first = parts[0] if parts else ''
            last = parts[-1] if len(parts) > 1 else ''
        
        return f"{first} {last}" if first or last else ''

    @staticmethod
    def _find_user_by_name_zip(
        first_name: str, 
        last_name: str, 
        zip_code: str, 
        users: Dict
    ) -> Optional[str]:
        # Delegate to global implementation
        try:
            return _find_user_by_name_zip_global(first_name, last_name, zip_code, users)
        except Exception:
            return None

    def apply_suggestions(
        self, 
        task: Dict[str, Any], 
        report: ValidationReport
    ) -> Dict[str, Any]:
        """Apply validation suggestions to fix task - consolidated version."""
        corrections = []
        
        if not isinstance(task, dict) or not isinstance(task.get('agt', []), list):
            return {"corrections": corrections}
        
        try:
            data, indexes = self._get_data_and_indexes()
            users = data.get('users', {})
            orders = data.get('orders', {})
            products = data.get('products', {})
            email_index = indexes['email_index']
            user_orders_index = indexes['user_orders_index']
            
            # Single-pass correction with context tracking
            current_user_id = None
            suggested_uid = self._extract_suggested_user_id(report)
            
            for action in task.get('agt', []):
                if not isinstance(action, dict):
                    continue

                # Normalize malformed action dicts that omit 'name' but include common keys
                # Examples: {'email': 'foo@example.com'} -> find_user_id_by_email
                #           {'order_number': 'W123'} -> get_order_details
                #           {'Street': '123 Market St', 'City': 'San Jose', ...} -> modify_pending_order_address
                if not action.get('name'):
                    # Top-level email
                    if 'email' in action and isinstance(action.get('email'), str):
                        action['name'] = 'find_user_id_by_email'
                        action['arguments'] = {'email': action.pop('email')}
                        # record as a correction-like metadata (we append to corrections below)
                    # Wrapper with 'arguments' containing email
                    elif isinstance(action.get('arguments'), dict) and 'email' in action.get('arguments'):
                        action['name'] = 'find_user_id_by_email'
                    # Top-level order number
                    elif 'order_number' in action or (isinstance(action.get('arguments'), dict) and 'order_number' in action.get('arguments')):
                        order_val = action.get('order_number') or action.get('arguments', {}).get('order_number')
                        action['name'] = 'get_order_details'
                        action['arguments'] = {'order_id': self._normalize_order_id(order_val)}
                    # Address-like dicts -> interpret as modify_pending_order_address
                    elif any(k.lower() in {'street','city','state','zip','zip_code','address1','address2'} for k in action.keys()):
                        args = action.get('arguments', {}) if isinstance(action.get('arguments'), dict) else {}
                        # collect address fields
                        addr = {}
                        for k in ['address1','address2','street','city','state','country','zip','zip_code']:
                            if k in action:
                                addr[k if k != 'street' else 'address1'] = action.pop(k)
                            elif isinstance(action.get('arguments'), dict) and k in action['arguments']:
                                addr[k if k != 'street' else 'address1'] = action['arguments'].pop(k)
                        action['name'] = 'modify_pending_order_address'
                        action['arguments'] = {**args, **addr}

                args = action.setdefault('arguments', {})
                action_name = action.get('name', '')

                # Update context
                if action_name in ['find_user_id_by_email', 'find_user_id_by_name_zip']:
                    if (user_id := args.get('user_id')) and user_id in users:
                        current_user_id = user_id
                
                # Apply user ID corrections
                if args.get('user_id') not in users:
                    # Try to extract missing name/zip from query if present (helps find_user_id_by_name_zip)
                    if action_name == 'find_user_id_by_name_zip':
                        # If name supplied as single string, split
                        name_val = args.get('name') or args.get('user_name') or None
                        if name_val and ('first_name' not in args or 'last_name' not in args):
                            parts = str(name_val).strip().split()
                            if len(parts) >= 2:
                                args.setdefault('first_name', parts[0])
                                args.setdefault('last_name', parts[-1])
                                corrections.append({'applied_to': action_name, 'fixed': 'split_name_into_first_last', 'new_value': {'first_name': parts[0], 'last_name': parts[-1]}})
                        # Try to extract zip from task query
                        if 'zip' not in args or not args.get('zip'):
                            qtext = task.get('q') or task.get('query') or ''
                            m = re.search(r"(\d{5})", qtext)
                            if m:
                                args['zip'] = m.group(1)
                                corrections.append({'applied_to': action_name, 'fixed': 'extracted_zip_from_query', 'new_value': args['zip']})

                    resolved_uid = (
                        self._resolve_from_email(args, users, email_index) or
                        self._resolve_from_name_zip(args, action, users) or
                        current_user_id or
                        suggested_uid or
                        next(iter(users.keys())) if users else None
                    )
                    
                    if resolved_uid:
                        old = args.get('user_id')
                        args['user_id'] = resolved_uid
                        corrections.append({
                            'applied_to': action_name,
                            'user_id': resolved_uid,
                            'old_value': old,
                            'reason': 'user_id_correction'
                        })
                
                # Apply order ID corrections
                if 'order_id' in args:
                    raw_oid = str(args['order_id'])
                    normalized = self._normalize_order_id(raw_oid)
                    replacement = self._find_replacement_order(
                        normalized, orders, user_orders_index, current_user_id
                    )
                    
                    if replacement and replacement != normalized:
                        old = args.get('order_id')
                        args['order_id'] = replacement
                        corrections.append({
                            'applied_to': action_name,
                            'fixed': 'order_id',
                            'old_value': old,
                            'new_value': replacement,
                            'reason': 'order_id_correction'
                        })
                
                # NEW: Apply item position/ID corrections
                if 'item_positions' in args or 'item_ids' in args:
                    self._fix_item_references(action, args, orders, corrections)

                # Ensure modify_pending_order_items has required fields
                if action_name == 'modify_pending_order_items':
                    # Ensure item_ids exists
                    item_ids = args.get('item_ids') or []
                    if not item_ids and 'order_id' in args:
                        # attempt to populate from order
                        oid = self._normalize_order_id(str(args.get('order_id')))
                        if oid in orders:
                            item_ids = [it.get('item_id') for it in orders[oid].get('items', []) if it.get('item_id')]
                            if item_ids:
                                args['item_ids'] = item_ids
                                corrections.append({
                                    'applied_to': action_name,
                                    'fixed': 'populated_item_ids_from_order',
                                    'new_value': item_ids
                                })
                    # Ensure new_item_ids exists
                    if 'new_item_ids' not in args or not args.get('new_item_ids'):
                        args['new_item_ids'] = list(item_ids)
                        corrections.append({
                            'applied_to': action_name,
                            'fixed': 'populated_new_item_ids',
                            'new_value': args['new_item_ids']
                        })
                    # Ensure payment_method_id exists (try to pick user's default)
                    if 'payment_method_id' not in args or not args.get('payment_method_id'):
                        uid = args.get('user_id') or current_user_id
                        payment_method = None
                        if uid and uid in users:
                            ums = users[uid].get('payment_methods', {})
                            payment_method = next(iter(ums.keys()), None)
                        args['payment_method_id'] = payment_method or 'ORIGINAL_PAYMENT_METHOD'
                        corrections.append({
                            'applied_to': action_name,
                            'fixed': 'populated_payment_method_id',
                            'new_value': args['payment_method_id']
                        })

                # Ensure modify_pending_order_payment has payment method and order_id
                if action_name == 'modify_pending_order_payment':
                    # Fill order_id if missing (use user's first order or prompt fallback)
                    if 'order_id' not in args or not args.get('order_id'):
                        candidate_oid = None
                        if current_user_id and current_user_id in user_orders_index and user_orders_index[current_user_id]:
                            candidate_oid = next(iter(user_orders_index[current_user_id]))
                        if not candidate_oid:
                            # fallback to any order in dataset
                            candidate_oid = next(iter(orders.keys())) if orders else None
                        if candidate_oid:
                            args['order_id'] = candidate_oid
                            corrections.append({
                                'applied_to': action_name,
                                'fixed': 'populated_order_id',
                                'new_value': args['order_id']
                            })
                    # Ensure payment_method_id exists (try to pick user's default)
                    if 'payment_method_id' not in args or not args.get('payment_method_id'):
                        uid = args.get('user_id') or current_user_id
                        payment_method = None
                        if uid and uid in users:
                            ums = users[uid].get('payment_methods', {})
                            payment_method = next(iter(ums.keys()), None)
                        args['payment_method_id'] = payment_method or 'ORIGINAL_PAYMENT_METHOD'
                        corrections.append({
                            'applied_to': action_name,
                            'fixed': 'populated_payment_method_id',
                            'new_value': args['payment_method_id']
                        })

                # If modifying a pending order address but no new address provided, attempt to infer one
                    addr_fields = {'address1', 'address2', 'city', 'state', 'country', 'zip'}
                    has_addr = all((k in args and args.get(k)) for k in addr_fields)
                    if not has_addr:
                        # Try to infer from order or user, else synthesize a simple new address
                        new_addr = {}
                        order_id = args.get('order_id') or args.get('order_ids')
                        if isinstance(order_id, list) and order_id:
                            order_id = order_id[0]
                        oid = self._normalize_order_id(str(order_id)) if order_id else None
                        if oid and oid in orders:
                            base_addr = orders[oid].get('address', {})
                        else:
                            # fallback to user's address if available
                            user_id = args.get('user_id') or current_user_id
                            base_addr = users.get(user_id, {}).get('address', {}) if user_id else {}

                        # Build new address using base fields where possible
                        new_addr['address1'] = args.get('address1') or base_addr.get('address1') or '123 New St'
                        new_addr['address2'] = args.get('address2') or base_addr.get('address2') or ''
                        new_addr['city'] = args.get('city') or base_addr.get('city') or base_addr.get('town') or 'San Jose'
                        new_addr['state'] = args.get('state') or base_addr.get('state') or base_addr.get('region') or 'CA'
                        new_addr['country'] = args.get('country') or base_addr.get('country') or 'USA'
                        new_addr['zip'] = args.get('zip') or base_addr.get('zip') or base_addr.get('zip_code') or '94101'

                        # Inject into args
                        for k, v in new_addr.items():
                            args[k] = v

                        corrections.append({
                            'applied_to': action_name,
                            'fixed': 'missing_address_fields',
                            'new_values': new_addr,
                            'reason': 'inferred_new_address_for_modify_action'
                        })
                        # Also inject a short explicit new_address statement into the query text
                        try:
                            current_q = task.get('q', '') or ''
                            address_text = f" Please change the shipping address to: {new_addr['address1']}, {new_addr['city']}, {new_addr['state']} {new_addr['zip']}."
                            if address_text.strip() not in current_q:
                                task['q'] = current_q + address_text
                        except Exception:
                            pass

                # If a return/exchange action is present but no item_ids specified, populate default items from order
                if action_name in ('return_delivered_order_items', 'exchange_delivered_order_items'):
                    if not args.get('item_ids') and (args.get('order_id') or args.get('order_ids')):
                        raw_oid = args.get('order_id') or (args.get('order_ids')[0] if isinstance(args.get('order_ids'), list) and args.get('order_ids') else None)
                        oid = self._normalize_order_id(str(raw_oid)) if raw_oid else None
                        if oid and oid in orders:
                            order_items = orders[oid].get('items', [])
                            default_item_ids = [item.get('item_id') for item in order_items if item.get('item_id')]
                            if default_item_ids:
                                args['item_ids'] = default_item_ids[:min(len(default_item_ids), 2)]
                                corrections.append({
                                    'applied_to': action_name,
                                    'fixed': 'item_ids_filled',
                                    'new_value': args['item_ids'],
                                    'reason': 'filled_missing_item_ids_from_order'
                                })
                                # Also append to query so the generated task explicitly contains these items
                                try:
                                    current_q = task.get('q', '') or ''
                                    items_str = ", ".join(args['item_ids'])
                                    q_text = f" The items to return/exchange are: {items_str}."
                                    if items_str and q_text.strip() not in current_q:
                                        task['q'] = current_q + q_text
                                except Exception:
                                    pass
            
            # Handle suggested actions from validation
            agt = task.setdefault('agt', [])
            for suggestion in getattr(report, 'suggestions', []) or []:
                if isinstance(suggestion, dict) and suggestion.get('suggest_action'):
                    suggested = suggestion.get('suggest_action')
                    if isinstance(suggested, dict) and suggested.get('name'):
                        agt.append(suggested)
                        corrections.append({
                            'applied_to': suggested.get('name'),
                            'action_added': suggested,
                            'reason': 'added_action_from_validation_suggestion'
                        })
            
            # CRITICAL FIX: Ensure authentication data is in query
            self._ensure_auth_data_in_query(task, users, corrections)
            
            return {"corrections": corrections}
            
        except Exception as e:
            logger.error(f"Failed to apply suggestions: {e}")
            return {"corrections": corrections}

    def _ensure_auth_data_in_query(
        self,
        task: Dict[str, Any],
        users: Dict,
        corrections: List[Dict[str, Any]]
    ) -> None:
        """Ensure query contains authentication data required by first action.
        
        This fixes the common issue where models output name-only responses
        because the training data doesn't include required auth fields.
        """
        try:
            actions = task.get('agt', [])
            if not actions:
                return
            
            first_action = actions[0]
            if not isinstance(first_action, dict):
                return
            
            action_name = first_action.get('name', '')
            args = first_action.get('arguments', {})
            query = task.get('q', '')
            
            # Handle find_user_id_by_email
            if action_name == 'find_user_id_by_email':
                email = args.get('email')
                if email and '@' in str(email):
                    # Check if email is in query
                    if email not in query and not re.search(r'[\w\.-]+@[\w\.-]+', query):
                        # Add email to query
                        additions = [
                            f"My email is {email}.",
                            f"My email address is {email}.",
                            f"I can be reached at {email}."
                        ]
                        addition = random.choice(additions)
                        task['q'] = query + " " + addition if query else addition
                        corrections.append({
                            'applied_to': 'query',
                            'fixed': 'added_email_to_query',
                            'value': email,
                            'reason': 'ensure_auth_data_in_query_for_find_user_id_by_email'
                        })
            
            # Handle find_user_id_by_name_zip
            elif action_name == 'find_user_id_by_name_zip':
                first_name = args.get('first_name')
                last_name = args.get('last_name')
                zip_code = args.get('zip')
                
                if first_name and last_name:
                    full_name = f"{first_name} {last_name}"
                    
                    # Check if name is in query
                    name_in_query = (
                        first_name.lower() in query.lower() and 
                        last_name.lower() in query.lower()
                    )
                    
                    if not name_in_query:
                        # Add name to query
                        additions = [
                            f"My name is {full_name}.",
                            f"This is {full_name}.",
                            f"Hi, I'm {full_name}."
                        ]
                        addition = random.choice(additions)
                        task['q'] = query + " " + addition if query else addition
                        query = task['q']
                        corrections.append({
                            'applied_to': 'query',
                            'fixed': 'added_name_to_query',
                            'value': full_name,
                            'reason': 'ensure_auth_data_in_query_for_find_user_id_by_name_zip'
                        })
                
                if zip_code and not re.search(r'\b\d{5}\b', query):
                    # Add zip to query
                    additions = [
                        f"My zip code is {zip_code}.",
                        f"I'm in zip code {zip_code}.",
                        f"My area code is {zip_code}."
                    ]
                    addition = random.choice(additions)
                    task['q'] = query + " " + addition if query else addition
                    corrections.append({
                        'applied_to': 'query',
                        'fixed': 'added_zip_to_query',
                        'value': zip_code,
                        'reason': 'ensure_auth_data_in_query_for_find_user_id_by_name_zip'
                    })
            
            # Handle find_user_id_by_username
            elif action_name == 'find_user_id_by_username':
                username = args.get('username')
                if username and username not in query:
                    # Add username to query
                    additions = [
                        f"My username is {username}.",
                        f"I'm user {username}.",
                        f"My account is {username}."
                    ]
                    addition = random.choice(additions)
                    task['q'] = query + " " + addition if query else addition
                    corrections.append({
                        'applied_to': 'query',
                        'fixed': 'added_username_to_query',
                        'value': username,
                        'reason': 'ensure_auth_data_in_query_for_find_user_id_by_username'
                    })
            
        except Exception as e:
            logger.warning(f"Failed to ensure auth data in query: {e}")

    def _find_replacement_order(
        self,
        original_oid: str,
        orders: Dict,
        user_orders_index: Dict[str, Set[str]],
        current_user_id: Optional[str]
    ) -> Optional[str]:
        """Find suitable replacement order."""
        # Order doesn't exist
        if original_oid not in orders:
            return next(iter(orders.keys())) if orders else None
        
        # Order exists but not accessible
        if (current_user_id and 
            current_user_id in user_orders_index and 
            original_oid not in user_orders_index[current_user_id] and
            user_orders_index[current_user_id]):
            return next(iter(user_orders_index[current_user_id]))
        
        return None  # No replacement needed
    
    def _fix_item_references(
        self,
        action: Dict[str, Any],
        args: Dict[str, Any],
        orders: Dict,
        corrections: List[Dict[str, Any]]
    ) -> None:
        """Fix invalid item_positions or item_ids to match order's actual items."""
        order_id = args.get('order_id')
        if not order_id:
            return
        
        normalized_oid = self._normalize_order_id(str(order_id))
        if normalized_oid not in orders:
            return
        
        order = orders[normalized_oid]
        order_items = order.get('items', [])
        
        if not isinstance(order_items, list) or not order_items:
            return
        
        # Fix item_positions
        if 'item_positions' in args:
            positions = args['item_positions']
            if isinstance(positions, list):
                max_position = len(order_items)
                # Filter out invalid positions
                valid_positions = [p for p in positions if 1 <= p <= max_position]
                
                if valid_positions != positions:
                    old = positions
                    args['item_positions'] = valid_positions or [1]  # Default to first item
                    corrections.append({
                        'applied_to': action.get('name'),
                        'fixed': 'item_positions',
                        'old_value': old,
                        'new_value': args['item_positions'],
                        'reason': f'Invalid positions for order with {max_position} items'
                    })
        
        # Fix item_ids
        if 'item_ids' in args:
            item_ids = args['item_ids']
            if isinstance(item_ids, list):
                valid_item_ids = {item.get('item_id') for item in order_items if isinstance(item, dict) and item.get('item_id')}
                # Filter out invalid IDs
                valid_ids = [iid for iid in item_ids if iid in valid_item_ids]
                
                if valid_ids != item_ids:
                    old = item_ids
                    # If no valid IDs, use first item's ID as fallback
                    if not valid_ids and valid_item_ids:
                        valid_ids = [next(iter(valid_item_ids))]
                    
                    args['item_ids'] = valid_ids
                    corrections.append({
                        'applied_to': action.get('name'),
                        'fixed': 'item_ids',
                        'old_value': old,
                        'new_value': args['item_ids'],
                        'reason': 'Invalid item IDs for this order'
                    })
                

    def _apply_business_logic_correction(
        self,
        action: Dict[str, Any],
        args: Dict[str, Any],
        action_name: str,
        orders: Dict,
        user_orders_index: Dict[str, Set[str]],
        current_user_id: Optional[str],
        corrections: List[Dict[str, Any]]
    ) -> None:
        """Attempt automatic corrections for actions that violate business rules.

        Examples:
        - If action is modify_pending_* but order is delivered, replace with return_delivered_order_items or escalate.
        - If action expects delivered order but order is pending, escalate or convert to transfer_to_human_agents.
        """
        if not isinstance(action, dict):
            return

        name = (action.get('name') or '').strip()
        if not name:
            return

        # Identify order ids referenced by this action
        order_ids = []
        if 'order_id' in args and args.get('order_id'):
            v = args.get('order_id')
            if isinstance(v, list):
                order_ids.extend(v)
            else:
                order_ids.append(v)
        if 'order_ids' in args and isinstance(args.get('order_ids'), list):
            order_ids.extend(args.get('order_ids'))

        pending_only = {
            'modify_pending_order_address',
            'modify_pending_order_payment',
            'modify_pending_order_items',
            'cancel_pending_order'
        }
        delivered_only = {
            'return_delivered_order_items',
            'exchange_delivered_order_items'
        }

        for raw_oid in order_ids:
            oid = self._normalize_order_id(str(raw_oid))
            order = orders.get(oid)
            if not order:
                continue
            status = str(order.get('status', '')).lower()

            # If action is pending-only but order is delivered -> convert
            if name in pending_only and status in ('delivered', 'completed', 'shipped'):
                old = name
                # Prefer an automated delivered-order action when possible
                new_name = 'return_delivered_order_items'
                # If replacement is intended, more complex logic may be needed; fallback to escalation
                if 'exchange' in old or 'modify_pending_order_items' in old:
                    new_name = 'exchange_delivered_order_items'

                action['name'] = new_name
                corrections.append({
                    'applied_to': old,
                    'old_name': old,
                    'new_name': new_name,
                    'reason': 'converted_pending_action_for_delivered_order',
                    'order_id': oid
                })

            # If action is delivered-only but order not delivered -> escalate
            if name in delivered_only and status not in ('delivered', 'completed', 'shipped'):
                old = name
                new_name = 'transfer_to_human_agents'
                action['name'] = new_name
                corrections.append({
                    'applied_to': old,
                    'old_name': old,
                    'new_name': new_name,
                    'reason': 'escalated_delivered_action_for_non_delivered_order',
                    'order_id': oid
                })

    def _resolve_from_email(self, args: Dict[str, Any], users: Dict, email_index: Dict[str, str]) -> Optional[str]:
        """Resolve user id from an email using the provided index."""
        email = args.get('email') if isinstance(args, dict) else None
        if not email or not isinstance(email, str):
            return None
        try:
            return email_index.get(email.lower().strip())
        except Exception:
            return None

    def _resolve_from_name_zip(self, args: Dict[str, Any], action: Dict[str, Any], users: Dict) -> Optional[str]:
        """Resolve user ID from name and zip."""
        try:
            if action.get('name') == 'find_user_id_by_name_zip':
                fn = args.get('first_name', '').strip()
                ln = args.get('last_name', '').strip()
                z = args.get('zip') or args.get('zip_code', '')
                if fn and ln and z:
                    return _find_user_by_name_zip_global(fn, ln, z, users)
        except Exception:
            pass
        return None

    def _extract_suggested_user_id(self, report: ValidationReport) -> Optional[str]:
        """Extract first suggested user ID from report."""
        for suggestion in report.suggestions:
            if 'suggest_user_id' in suggestion:
                return suggestion['suggest_user_id']
        return None

@dataclass
class TaskGenerationMemory:
    """
    Memory system for task generation following AgentFlow architecture.
    Tracks state across generation turns (M^t -> M^t+1)
    
    This extends the general Memory concept to task generation specifically,
    maintaining context about what has been generated, validated, and refined.
    """
    turn: int
    scenario: str  # The scenario being developed
    prompt_data: Dict[str, Any]  # Knowledge base (K): users, orders, tools, policies
    
    # Generation state
    current_sub_goal: Dict[str, Any] = field(default_factory=dict)
    planned_components: List[Dict[str, Any]] = field(default_factory=list)  # Components to generate
    generated_components: Dict[str, Any] = field(default_factory=dict)  # Generated task parts
    validation_results: List[ValidationReport] = field(default_factory=list)  # Validation history
    corrections_applied: List[Dict[str, Any]] = field(default_factory=list)  # Correction history
    # Full verifier outputs (including status and next_action) for planner to inspect
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    # Track per-component attempt counts to avoid infinite retries
    component_attempts: Dict[str, int] = field(default_factory=dict)
    # Execution-time diagnostics/errors collected when executor fails
    execution_errors: List[Dict[str, Any]] = field(default_factory=list)
    # Track replan attempts per component to avoid flapping
    replan_attempts: Dict[str, int] = field(default_factory=dict)
    
    # Task assembly state
    query: str = ""  # User query (q)
    actions: List[Dict[str, Any]] = field(default_factory=list)  # Actions (agt)
    outputs: List[str] = field(default_factory=list)  # Outputs (ogt)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary - simplified serialization."""
        # Convert ValidationReport objects to dicts
        serialized_validation = [
            {'valid': v.valid, 'missing': v.missing, 'suggestions': v.suggestions}
            if isinstance(v, ValidationReport) else v
            for v in self.validation_results
        ]

        # Simplify verification history (avoid deep recursion)
        serialized_verification = []
        for entry in self.verification_history:
            if isinstance(entry, dict):
                simple_entry = {
                    'status': entry.get('verification_status') or entry.get('status'),
                    'next_action': entry.get('next_action')
                }
                if 'validation_report' in entry:
                    vr = entry['validation_report']
                    if isinstance(vr, ValidationReport):
                        simple_entry['valid'] = vr.valid
                        simple_entry['missing_count'] = len(vr.missing)
                serialized_verification.append(simple_entry)

        return {
            'turn': self.turn,
            'scenario': self.scenario,
            'current_sub_goal': self.current_sub_goal,
            'query': self.query,
            'actions': self.actions,
            'outputs': self.outputs,
            'validation_results': serialized_validation,
            'verification_history': serialized_verification,
            'component_attempts': self.component_attempts,
            'corrections_count': len(self.corrections_applied)
        }
    
    def get_context_summary(self) -> str:
        """Get summary for next turn"""
        summary_parts = []
        
        if self.query:
            summary_parts.append(f"Query generated: {len(self.query)} chars")
        if self.actions:
            summary_parts.append(f"Actions: {len(self.actions)} planned")
        if self.validation_results:
            last_validation = self.validation_results[-1]
            summary_parts.append(f"Last validation: {'✓ valid' if last_validation.valid else '✗ invalid'}")
        if self.corrections_applied:
            summary_parts.append(f"Corrections: {len(self.corrections_applied)} applied")

        # Include last verification brief
        if self.verification_history:
            last_ver = self.verification_history[-1]
            status = last_ver.get('verification_status') or last_ver.get('status')
            next_action = last_ver.get('next_action')
            summary_parts.append(f"Last verification: {status} -> {next_action}")

        # Include attempts summary
        if self.component_attempts:
            attempts_summary = ", ".join([f"{k}:{v}" for k, v in self.component_attempts.items()])
            summary_parts.append(f"Attempts: {attempts_summary}")
        
        return " | ".join(summary_parts)
    
    def clone_for_next_turn(self) -> 'TaskGenerationMemory':
        """Create memory for next turn"""
        return TaskGenerationMemory(
            turn=self.turn + 1,
            scenario=self.scenario,
            prompt_data=self.prompt_data,
            current_sub_goal=self.current_sub_goal.copy() if self.current_sub_goal else {},
            planned_components=self.planned_components.copy(),
            generated_components=self.generated_components.copy(),
            validation_results=self.validation_results.copy(),
            corrections_applied=self.corrections_applied.copy(),
            verification_history=self.verification_history.copy(),
            component_attempts=self.component_attempts.copy(),
            query=self.query,
            actions=self.actions.copy(),
            outputs=self.outputs.copy()
        )


class TaskGenerationPlanner:
    """
    Planner for task generation: Decides what components to generate next
    
    Input: Scenario, Current Memory (M^t), Prompt Data (K)
    Output: Sub-goal (component to generate), Approach (a^t)
    """
    
    def __init__(self, client: OpenAI, config: TauBenchConfig):
        self.client = client
        self.config = config
    
    def _assess_scenario_complexity(self, query: str) -> str:
        """Assess query complexity for adaptive generation."""
        query_lower = query.lower()
        
        # Count operations
        operations = sum([
            query_lower.count('return'),
            query_lower.count('exchange'),
            query_lower.count('modify'),
            query_lower.count('cancel'),
            query_lower.count('refund'),
            query_lower.count('update'),
            query_lower.count('change'),
            query_lower.count('replace')
        ])
        
        # Count entities (order IDs)
        import re
        entities = len(re.findall(r'#?W?\d{6,8}', query))
        
        # Count items/problems mentioned
        items = sum([
            query_lower.count('item'),
            query_lower.count('product'),
            query_lower.count('order'),
            query_lower.count('wrong'),
            query_lower.count('broken'),
            query_lower.count('damaged'),
            query_lower.count('missing')
        ])
        
        complexity_score = operations + entities * 2 + items * 0.5
        
        if complexity_score > 6:
            return 'complex'
        elif complexity_score > 3:
            return 'medium'
        return 'simple'
    
    def plan(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Plan next generation step"""
        # Determine what to generate based on current state
        # Use verifier signals and per-component attempt counts to make robust decisions
        # Prefer using explicit verifier signals (if available) for finer decisions
        if memory.verification_history:
            last_ver = memory.verification_history[-1]
            next_action = last_ver.get('next_action')
            status = last_ver.get('verification_status') or last_ver.get('status')
            validation_report = last_ver.get('validation_report') or {}

            # If validator suggested adding actions because OGT promised outcomes,
            # instruct planner to regenerate actions and include required action names.
            try:
                suggestions = []
                if isinstance(validation_report, ValidationReport):
                    suggestions = validation_report.suggestions or []
                elif isinstance(validation_report, dict):
                    suggestions = validation_report.get('suggestions', []) or []

                required_actions = set()
                for s in suggestions:
                    if not isinstance(s, dict):
                        continue
                    # suggestion may come from _validate_q_action_consistency
                    if s.get('type') == 'q_action_mismatch' or s.get('missing_actions'):
                        if s.get('missing_actions'):
                            required_actions.update(s.get('missing_actions', []))
                        elif s.get('suggest_action') and isinstance(s.get('suggest_action'), dict):
                            required_actions.add(s['suggest_action'].get('name'))

                if required_actions:
                    return {
                        'sub_goal': {
                            'component': 'actions',
                            'description': 'Regenerate actions (include required tools from Q suggestions)',
                            'required_actions': list(required_actions)
                        },
                        'approach': 'regenerate_actions_with_required_tools',
                        'reasoning': 'Verifier suggested required actions to match Q promises'
                    }
            except Exception:
                # If anything goes wrong here, fall back to existing logic
                pass
            # Use a small retry budget per component to avoid tight loops
            max_component_retries = getattr(self.config, 'agentflow_component_retries', 3)

            # If verifier explicitly asks to apply corrections, do that first
            if next_action == 'apply_corrections' or (status == 'needs_correction'):
                return {
                    'sub_goal': {'component': 'corrections', 'description': 'Apply validation corrections'},
                    'approach': 'fix_validation_issues',
                    'reasoning': 'Verifier requested corrections based on validation suggestions'
                }
            
            # NEW: Handle Q-AGT mismatch by adding missing actions
            if next_action == 'add_missing_actions' or (status == 'needs_action_fix'):
                return {
                    'sub_goal': {'component': 'corrections', 'description': 'Add missing actions from Q validation'},
                    'approach': 'add_required_actions',
                    'reasoning': 'Q promises outcomes not implemented in AGT - adding required actions'
                }

            # If verifier requests a retry, attempt to infer which component to retry
            if next_action == 'retry':
                if isinstance(validation_report, ValidationReport):
                    missing = validation_report.missing or []
                elif isinstance(validation_report, dict):
                    missing = validation_report.get('missing', [])
                else:
                    missing = []

                missing_types = {m.get('type') for m in missing if isinstance(m, dict)}

                # Decide which component to retry based on missing types
                if any(t in missing_types for t in ('order_id', 'product_id')):
                    comp = 'actions'
                elif any(t in missing_types for t in ('email', 'user_id', 'incomplete_name_zip', 'name_zip_mismatch')):
                    comp = 'corrections'
                else:
                    comp = 'actions'

                attempts = memory.component_attempts.get(comp, 0)
                if attempts >= max_component_retries:
                    # If we've retried enough, fall back to corrections to salvage the task
                    return {
                        'sub_goal': {'component': 'corrections', 'description': 'Apply validation corrections'},
                        'approach': 'fix_validation_issues',
                        'reasoning': f'Max retries reached for {comp} ({attempts}) - applying corrections'
                    }

                # Otherwise retry the selected component
                if comp == 'actions':
                    return {
                        'sub_goal': {'component': 'actions', 'description': 'Regenerate actions (agt)'},
                        'approach': 'regenerate_actions_for_query',
                        'reasoning': 'Order/product references missing or invalid - regenerate actions'
                    }
                else:
                    return {
                        'sub_goal': {'component': 'corrections', 'description': 'Apply validation corrections'},
                        'approach': 'fix_validation_issues',
                        'reasoning': 'Verifier requested retry - applying corrections first'
                    }

        if not memory.query:
            return {
                'sub_goal': {'component': 'query', 'description': 'Generate user query (q)'},
                'approach': 'generate_from_scenario',
                'reasoning': 'Query is the foundation - generate it first'
            }
        
        if not memory.actions:
            # Assess query complexity for adaptive generation
            complexity = self._assess_scenario_complexity(memory.query) if memory.query else 'simple'
            
            reasoning = f'Query exists (complexity: {complexity}), now plan required actions'
            if complexity == 'complex':
                reasoning += ' - Breaking down into simpler steps'
            
            return {
                'sub_goal': {
                    'component': 'actions',
                    'description': f'Generate expected actions (agt) - {complexity} scenario',
                    'complexity': complexity
                },
                'approach': 'plan_actions_for_query',
                'reasoning': reasoning
            }
        
        # If no explicit verifier signal was available earlier, fallback to older pattern
        if memory.validation_results and not memory.validation_results[-1].valid:
            return {
                'sub_goal': {'component': 'corrections', 'description': 'Apply validation corrections'},
                'approach': 'fix_validation_issues',
                'reasoning': 'Validation failed, need to apply corrections'
            }
        
        if not memory.outputs:
            return {
                'sub_goal': {'component': 'outputs', 'description': 'Generate expected outputs (ogt)'},
                'approach': 'synthesize_outputs',
                'reasoning': 'Query and actions exist, synthesize expected outputs'
            }
        
        # All components generated and validated
        return {
            'sub_goal': {'component': 'complete', 'description': 'Task generation complete'},
            'approach': 'finalize',
            'reasoning': 'All components generated and validated'
        }

    def replan_with_diagnostics(self, memory: TaskGenerationMemory, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Replanning hook invoked when the executor failed.

        The diagnostics dict is expected to contain at least:
          - component: which component failed (query/actions/outputs/corrections)
          - error: textual error or exception
          - result: raw executor result dict (may contain partial outputs)

        This method decides whether to: regenerate the same component, escalate to
        regenerate the query, or apply corrections first. It uses per-component
        replan attempt counters to avoid infinite loops.
        """
        comp = diagnostics.get('component') or diagnostics.get('failed_component') or 'unknown'
        error = diagnostics.get('error', '')

        # Basic guard: limit replans per component
        max_replans = getattr(self.config, 'agentflow_max_replans', 2)
        attempts = memory.replan_attempts.get(comp, 0)
        memory.replan_attempts[comp] = attempts + 1

        if attempts >= max_replans:
            # If we've replanned enough, fall back to applying corrections or giving up
            return {
                'sub_goal': {'component': 'corrections', 'description': 'Apply validation corrections'},
                'approach': 'fix_validation_issues',
                'reasoning': f'Max replan attempts reached for {comp} ({attempts}) - applying corrections'
            }

        # Heuristic rules based on failure component and error content
        if comp == 'actions':
            # If actions generation failed due to missing references, try regenerating query
            if 'order' in str(error).lower() or 'missing' in str(error).lower() or diagnostics.get('partial'):
                return {
                    'sub_goal': {'component': 'query', 'description': 'Regenerate query to provide clearer context'},
                    'approach': 'regenerate_query_with_more_context',
                    'reasoning': 'Actions failed - regenerating query to include clearer order/user context'
                }
            # Otherwise, try regenerating actions with a different approach
            return {
                'sub_goal': {'component': 'actions', 'description': 'Regenerate actions (agt)'},
                'approach': 'regenerate_actions_with_alternative_constraints',
                'reasoning': 'Retrying actions generation with adjusted constraints'
            }

        if comp == 'query':
            # If query generation failed, fallback to a simpler template-based query
            return {
                'sub_goal': {'component': 'query', 'description': 'Generate a simpler query using a template'},
                'approach': 'generate_simple_query_template',
                'reasoning': 'Query generation failed - use a deterministic template to recover'
            }

        if comp == 'outputs':
            # Outputs are optional; try generating outputs deterministically
            return {
                'sub_goal': {'component': 'outputs', 'description': 'Generate outputs deterministically'},
                'approach': 'synthesize_outputs_deterministic',
                'reasoning': 'Outputs generation failed - attempt deterministic synthesis'
            }

        # Default fallback: ask to regenerate actions
        return {
            'sub_goal': {'component': 'actions', 'description': 'Regenerate actions (agt)'},
            'approach': 'regenerate_actions_general',
            'reasoning': 'Executor failed - default to regenerating actions'
        }


class TaskGenerationExecutor:
    """
    Executor for task generation: Generates task components
    
    Input: Plan, Memory, Prompt Data
    Output: Generated Component (e^t)
    """
    
    def __init__(self, client: OpenAI, config: TauBenchConfig, data_reader: TauBenchDataReader):
        self.client = client
        self.config = config
        self.data_reader = data_reader
        self._tool_schemas = None  # Cache for tool schemas
    
    def _load_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load and cache tool schemas (delegates to module-level helper)."""
        if self._tool_schemas is None:
            try:
                self._tool_schemas = _get_tool_schemas(self.data_reader)
            except Exception as e:
                logger.warning(f"Failed to load tool schemas via helper: {e}")
                self._tool_schemas = {}

        return self._tool_schemas
    
    def _validate_action_arguments(self, action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate action arguments against tool schema."""
        name = action.get('name', '')
        args = action.get('arguments', {})
        
        schemas = self._load_tool_schemas()
        if name not in schemas:
            # Unknown tool - let validator catch this
            return True, None
        
        schema = schemas[name]
        required = schema.get('required', [])
        
        # Check required parameters
        missing = [p for p in required if p not in args]
        if missing:
            return False, f"Missing required parameters for {name}: {missing}"
        
        # Check for complex nested objects (heuristic)
        for key, value in args.items():
            if isinstance(value, dict):
                # Check if it's a simple dict or complex nested structure
                if any(isinstance(v, (dict, list)) for v in value.values()):
                    return False, f"Invalid nested object in {key} - use simple formats"
            elif isinstance(value, list):
                # Check if list contains complex objects
                if value and isinstance(value[0], dict):
                    # Allow simple dicts, reject complex ones
                    if any(isinstance(v, (dict, list)) for item in value if isinstance(item, dict) for v in item.values()):
                        return False, f"Invalid nested objects in {key} - use simple IDs or positions"
        
        return True, None
    
    def execute(self, plan: Dict[str, Any], memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Execute generation for planned component"""
        
        component = plan['sub_goal']['component']
        approach = plan['approach']
        
        if component == 'query':
            return self._generate_query(memory)
        elif component == 'actions':
            return self._generate_actions(memory)
        elif component == 'outputs':
            return self._generate_outputs(memory)
        elif component == 'corrections':
            return self._apply_corrections(memory)
        else:
            return {
                'success': False,
                'error': f'Unknown component: {component}'
            }
    
    def _generate_query(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Generate user query from scenario and data"""
        
        # Build prompt for query generation
        prompt = self._build_query_prompt(memory)
        
        try:
            # Use APIClient.call_with_retry to get robust retry/backoff behavior
            api_client = APIClient(self.client, self.config)
            try:
                content = api_client.call_with_retry(prompt)
            except Exception as e:
                logger.error(f"Query generation failed (API): {e}")
                return {'success': False, 'error': str(e)}

            if not content:
                logger.error("Query generation failed: empty response content")
                return {'success': False, 'error': 'Empty content from LLM'}

            query = content.strip()
            # Remove any wrapping quotes or formatting
            query = query.strip('"\'`')
            # Ensure query references a real product from sampled orders if applicable
            try:
                sampled_products = memory.prompt_data.get('sampled_products', {}) or {}
                # build simple token sets
                product_names = {str(p.get('name')).lower() for p in sampled_products.values() if isinstance(p, dict) and p.get('name')}
                product_ids = set(sampled_products.keys())
                found = False
                if product_names:
                    q_lower = query.lower()
                    for pn in product_names:
                        if pn and pn in q_lower:
                            found = True
                            break
                if not found and product_ids:
                    for pid in product_ids:
                        if str(pid) in query:
                            found = True
                            break
                if not found:
                    # Append first product info to the query to improve grounding
                    # Prefer an item from the first sampled order
                    orders = memory.prompt_data.get('sampled_orders', []) or []
                    if orders and isinstance(orders[0], dict):
                        first_items = orders[0].get('items') or []
                        if isinstance(first_items, list) and first_items:
                            it = first_items[0]
                            pid = it.get('product_id') or it.get('pid') or it.get('product') or it.get('sku')
                            pname = None
                            if isinstance(it.get('product_details'), dict):
                                pname = it.get('product_details').get('name')
                            if not pname and pid and str(pid) in sampled_products:
                                p = sampled_products.get(str(pid))
                                pname = p.get('name') if isinstance(p, dict) else p
                            if pid or pname:
                                append_piece = f" This concerns the {pname or 'product'} (product_id: {pid or 'Unknown'})."
                                query = query.strip() + append_piece
            except Exception:
                # Don't let postprocessing step break generation
                pass
            
            return {
                'success': True,
                'query': query,
                'reasoning': f'Generated query from scenario: {memory.scenario[:50]}'
            }
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_query_prompt(self, memory: TaskGenerationMemory) -> str:
        """Build prompt for query generation"""
        prompt_parts = []
        prompt_parts.append(f"**Scenario:** {memory.scenario}")

        # Add user data context (concise)
        user_data = memory.prompt_data.get('sampled_user_details', {})
        if user_data:
            prompt_parts.append(f"\n**User (use these exact details):**")
            name = user_data.get('name', 'Unknown')
            email = user_data.get('email', 'Unknown')
            prompt_parts.append(f"- Name: {name}")
            prompt_parts.append(f"- Email: {email}")
            addr = user_data.get('address')
            if isinstance(addr, dict):
                prompt_parts.append(f"- City: {addr.get('city', 'Unknown')}, Zip: {addr.get('zip', 'Unknown')}")

        # Add order context if available (include multiple orders to allow multi-step scenarios)
        orders = memory.prompt_data.get('sampled_orders', [])
        if orders:
            prompt_parts.append(f"\n**Order Context (use exact IDs and product info if referenced):**")
            sampled_products = memory.prompt_data.get('sampled_products', {}) or {}
            for order in orders[:3]:
                if isinstance(order, dict):
                    o_items = order.get('items', []) if isinstance(order.get('items', []), list) else []
                    items_count = len(o_items)
                    prompt_parts.append(f"- {order.get('order_id', 'Unknown')}: status={order.get('status', 'Unknown')}, items={items_count}")
                    # Include product names/ids for up to 3 items per order
                    if o_items:
                        item_summaries = []
                        for item in o_items[:3]:
                            if not isinstance(item, dict):
                                continue
                            pid = item.get('product_id') or item.get('pid') or item.get('product') or item.get('sku')
                            pname = None
                            if isinstance(item.get('product_details'), dict):
                                pname = item.get('product_details', {}).get('name')
                            if not pname and pid and str(pid) in sampled_products:
                                prod = sampled_products.get(str(pid))
                                pname = prod.get('name') if isinstance(prod, dict) else prod
                            qty = item.get('quantity') or item.get('qty') or 1
                            pid_str = str(pid) if pid else 'Unknown'
                            item_summaries.append(f"{qty}x {pname or 'Unknown'} ({pid_str})")
                        if item_summaries:
                            prompt_parts.append(f"  Items detail: {', '.join(item_summaries)}")

        # Add persona and policy hints
        personas = memory.prompt_data.get('sampled_personas', [])
        if personas:
            prompt_parts.append(f"\n**Persona:** {personas[0]}")

        sampled_policy = memory.prompt_data.get('sampled_policy', {})
        if sampled_policy:
            prompt_parts.append("\n**Policy Constraints (observe these):**")
            # include a short summary if possible
            try:
                pol_keys = list(sampled_policy.keys())[:3]
                prompt_parts.append(f"- Policies: {', '.join(pol_keys)}")
            except Exception:
                pass

        # Request a more complex, multi-step natural query
        prompt_parts.append("\n**Task:** Generate a NATURAL customer query (single paragraph, conversational) that meets ALL of the following:")
        prompt_parts.append("1) Is realistic and matches the Scenario and Persona above.")
        prompt_parts.append("2) Uses the actual user's name/email/details provided above (do NOT invent other user identities).")
        prompt_parts.append("2b) If you reference specific products/items in the query, use the EXACT product names and product_id values shown in the Order Context above (do NOT invent product names or IDs).")
        prompt_parts.append("3) Explicitly references one or more exact order IDs from the Order Context when relevant.")
        prompt_parts.append("4) Implies a multi-step workflow the agent must perform (for example: authenticate -> get_order_details -> modify_order -> issue_refund -> complete). The query should make these needs clear without writing out tool calls.")
        prompt_parts.append("5) May include conditional preferences or constraints (e.g., 'only refund to original payment method', 'ship to my new address if replacement is issued').")
        prompt_parts.append("6) Expresses urgency, tone, and desired outcome (e.g., refund, replacement, expedited shipping) in natural language.")
        prompt_parts.append("7) Keep it realistic (short-to-medium length, 1-3 sentences) and output ONLY the query text — no JSON, no tags, no extra commentary.")

        # Guidance to encourage complexity and diversity
        prompt_parts.append("\nHints for complexity (do NOT output these hints):")
        prompt_parts.append("- Consider combining requests (e.g., return + refund + address update) or conditional asks")
        prompt_parts.append("- Mention exact order IDs and items where appropriate")
        prompt_parts.append("- Indicate the reason/condition for the request (damaged, wrong item, late delivery)")
        prompt_parts.append("- Use a natural tone matching the persona (polite, frustrated, urgent, etc.)")

        return "\n".join(prompt_parts)
    
    def _generate_actions(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Generate expected actions for the query"""
        
        prompt = self._build_actions_prompt(memory)
        
        try:
            # Use APIClient for retries and robust extraction
            api_client = APIClient(self.client, self.config)
            try:
                content = api_client.call_with_retry(prompt)
            except Exception as e:
                logger.error(f"Actions generation failed (API): {e}")
                return {'success': False, 'error': str(e), 'actions': []}

            if not content:
                logger.error("Actions generation failed: empty response content")
                return {'success': False, 'error': 'Empty content from LLM', 'actions': []}

            # Parse actions from response
            actions = self._parse_actions(content)
            
            return {
                'success': True,
                'actions': actions,
                'reasoning': f'Generated {len(actions)} actions for query'
            }
            
        except Exception as e:
            logger.error(f"Actions generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_actions_prompt(self, memory: TaskGenerationMemory) -> str:
        """Build prompt for actions generation"""
        
        prompt_parts = []
        prompt_parts.append(f"**User Query:** {memory.query}")
        
        # Add available tools
        tools = memory.prompt_data.get('sampled_api', {}).get('all_tools', [])
        if tools:
            tool_names = [t.get('function', {}).get('name', '') for t in tools]
            prompt_parts.append(f"\n**Available Tools:** {', '.join(tool_names)}")
        
        # Add user context
        user_data = memory.prompt_data.get('sampled_user_details', {})
        if user_data:
            prompt_parts.append(f"\n**User Context:**")
            prompt_parts.append(f"- User ID: {user_data.get('user_id', 'Unknown')}")
            prompt_parts.append(f"- Name: {user_data.get('name', 'Unknown')}")
            prompt_parts.append(f"- Email: {user_data.get('email', 'Unknown')}")
        
        # Add order context
        orders = memory.prompt_data.get('sampled_orders', [])
        if orders:
            prompt_parts.append(f"\n**Order Context:**")
            sampled_products = memory.prompt_data.get('sampled_products', {}) or {}
            for order in orders[:2]:
                if isinstance(order, dict):
                    prompt_parts.append(f"- Order {order.get('order_id', 'Unknown')}: {order.get('status', 'Unknown')}")
                    o_items = order.get('items', []) if isinstance(order.get('items', []), list) else []
                    if o_items:
                        item_summaries = []
                        for item in o_items[:3]:
                            if not isinstance(item, dict):
                                continue
                            pid = item.get('product_id') or item.get('pid') or item.get('product') or item.get('sku')
                            pname = None
                            if isinstance(item.get('product_details'), dict):
                                pname = item.get('product_details', {}).get('name')
                            if not pname and pid and str(pid) in sampled_products:
                                prod = sampled_products.get(str(pid))
                                pname = prod.get('name') if isinstance(prod, dict) else prod
                            qty = item.get('quantity') or item.get('qty') or 1
                            pid_str = str(pid) if pid else 'Unknown'
                            item_summaries.append(f"{qty}x {pname or 'Unknown'} ({pid_str})")
                        if item_summaries:
                            prompt_parts.append(f"  Items detail: {', '.join(item_summaries)}")

        # If the planner requested specific actions, include them as MUST-have constraints
        try:
            sub = getattr(memory, 'current_sub_goal', None) or {}
            req = sub.get('required_actions') if isinstance(sub, dict) else None
            if req:
                req_list = [r for r in req if isinstance(r, str) and r]
                if req_list:
                    prompt_parts.append("\n\nCONSTRAINT: The generated actions MUST include the following tool names:")
                    prompt_parts.append(", ".join(req_list))
                    prompt_parts.append("Also ensure each required action includes fully populated arguments (user_id, order_id, items) using the exact values from the User Data and Order Data sections.")
        except Exception:
            pass
        
        prompt_parts.append("\n**Task:** Generate a list of actions (tool calls) needed to fulfill this query.")
        prompt_parts.append("Format as JSON array:")
        prompt_parts.append('[')
        prompt_parts.append('  {"name": "tool_name", "arguments": {...}},')
        prompt_parts.append('  ...')
        prompt_parts.append(']')
        
        prompt_parts.append("\n**CRITICAL RULES (MUST FOLLOW):**")
        
        # Rule 1: MANDATORY AUTHENTICATION (based on what's in the QUERY, not just user data)
        prompt_parts.append("\n1. **AUTHENTICATION IS MANDATORY - Use method matching query content**")
        
        # Detect what auth info is actually in the query
        query_text = memory.query or ''
        has_email_in_query = '@' in query_text
        has_name_in_query = user_data and user_data.get('name') and user_data.get('name') in query_text
        has_zip_in_query = user_data and user_data.get('zip') and str(user_data.get('zip')) in query_text
        
        if has_email_in_query and user_data and user_data.get('email'):
            # Email is in the query - use email auth
            prompt_parts.append("   ✓ The query mentions an email address. The FIRST action MUST be:")
            prompt_parts.append("   ```json")
            prompt_parts.append("   {")
            prompt_parts.append('     "name": "find_user_id_by_email",')
            prompt_parts.append('     "arguments": {')
            prompt_parts.append(f'       "email": "{user_data.get("email")}"')
            prompt_parts.append("     }")
            prompt_parts.append("   }")
            prompt_parts.append("   ```")
        elif (has_name_in_query or (has_zip_in_query and user_data and user_data.get('first_name'))) and user_data:
            # Name and/or zip in query - use name+zip auth
            prompt_parts.append("   ✓ The query mentions a name and/or zip code. The FIRST action MUST be:")
            prompt_parts.append("   ```json")
            prompt_parts.append("   {")
            prompt_parts.append('     "name": "find_user_id_by_name_zip",')
            prompt_parts.append('     "arguments": {')
            if user_data.get('first_name'):
                prompt_parts.append(f'       "first_name": "{user_data.get("first_name")}",')
            if user_data.get('last_name'):
                prompt_parts.append(f'       "last_name": "{user_data.get("last_name")}",')
            if user_data.get('zip'):
                prompt_parts.append(f'       "zip": "{user_data.get("zip")}"')
            prompt_parts.append("     }")
            prompt_parts.append("   }")
            prompt_parts.append("   ```")
        else:
            # No explicit auth info in query - must infer from available data
            prompt_parts.append("   ✓ Choose the appropriate authentication method based on available query information.")
            if user_data:
                prompt_parts.append("     Available options:")
                if user_data.get('email'):
                    prompt_parts.append(f'     - Email: {user_data.get("email")}')
                if user_data.get('first_name') and user_data.get('last_name') and user_data.get('zip'):
                    prompt_parts.append(f'     - Name + Zip: {user_data.get("first_name")} {user_data.get("last_name")}, {user_data.get("zip")}')
        
        # Rule 2: Use exact IDs
        prompt_parts.append("\n2. Use ONLY the available tools listed above")
        prompt_parts.append("\n2b. If you reference specific products/items in actions, use the EXACT product names and product_id values shown in the Order Context above (do NOT invent product names or IDs).")
        prompt_parts.append("3. Include user_id in ALL actions that need it (from User Context above)")
        prompt_parts.append("4. Use exact order_ids from the Order Context (e.g., #W1234567)")
        
        # Rule 3: Argument format constraints
        prompt_parts.append("\n5. **ARGUMENT FORMAT RULES:**")
        prompt_parts.append("   - Use SIMPLE formats: strings, numbers, booleans, simple arrays")
        prompt_parts.append("   - For item references: use item_ids (array of strings) or item_positions (array of integers)")
        prompt_parts.append("   - For addresses: use simple string format unless tool specifically requires object")
        prompt_parts.append("   - For boolean flags: use true/false (e.g., refund_to_original_payment_method: true)")
        prompt_parts.append("   - DO NOT invent complex nested objects with descriptions/quantities")
        
        # NEW: Rule for item reference validation
        prompt_parts.append("\n5b. **ITEM REFERENCE VALIDATION:**")
        if orders:
            prompt_parts.append("   **CRITICAL:** Validate item references against actual order contents:")
            for order in orders[:2]:
                if isinstance(order, dict):
                    oid = order.get('order_id', 'Unknown')
                    items = order.get('items', [])
                    if isinstance(items, list) and items:
                        prompt_parts.append(f"   - Order {oid} has {len(items)} items (valid positions: 1 to {len(items)})")
                        # Show item IDs if available
                        item_ids = [item.get('item_id') for item in items if isinstance(item, dict) and item.get('item_id')]
                        if item_ids:
                            sampled_products = memory.prompt_data.get('sampled_products', {}) or {}
                            # build mapping for display
                            display_items = []
                            for iid in item_ids[:3]:
                                # find product name
                                prod_name = None
                                # search items for matching item_id to fetch product name
                                for it in items:
                                    if isinstance(it, dict) and (it.get('item_id') == iid or it.get('id') == iid):
                                        pd = it.get('product_details') or {}
                                        prod_name = pd.get('name') if isinstance(pd, dict) else None
                                        if not prod_name:
                                            pid = it.get('product_id') or it.get('pid') or it.get('product') or it.get('sku')
                                            if pid and str(pid) in sampled_products:
                                                p = sampled_products.get(str(pid))
                                                prod_name = p.get('name') if isinstance(p, dict) else p
                                        break
                                display_items.append({"item_id": iid, "product": prod_name or "Unknown"})
                            prompt_parts.append(f"     Valid item_ids: {display_items}{'...' if len(item_ids) > 3 else ''}")
            prompt_parts.append("   **MUST:** Only reference positions/IDs that actually exist in the order")
        
        # Rule 4: Action selection
        prompt_parts.append("\n6. **ACTION SELECTION:**")
        prompt_parts.append("   - For wrong items delivered (wrong product): if the user asks for refund/return, use return_delivered_order_items; if the user requests a replacement/exchange, use exchange_delivered_order_items")
        prompt_parts.append("   - For damaged/broken items: prefer exchange_delivered_order_items if user asks for replacement; if user asks for refund, use return_delivered_order_items")
        prompt_parts.append("   - For address changes when the order is still pending: use modify_pending_order_address.")
        prompt_parts.append("   - Follow logical order: authenticate -> get info -> perform action. Execute verified modifications immediately after authentication and verification.")
        
        return "\n".join(prompt_parts)
    
    def _parse_actions(self, content: str) -> List[Dict[str, Any]]:
        """Parse and validate actions from LLM response."""
        try:
            # Try to find JSON array in content (fast path)
            actions = None
            if '[' in content and ']' in content:
                try:
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    json_str = content[start:end]
                    actions = json.loads(json_str)
                except Exception:
                    actions = None

            # If fast path failed, try the centralized JSON candidate extractor
            if not isinstance(actions, list):
                parsed = _extract_json_candidate_from_text(content)
                if isinstance(parsed, list):
                    actions = parsed
                elif isinstance(parsed, dict) and parsed.get('agt') and isinstance(parsed.get('agt'), list):
                    actions = parsed.get('agt')

            if isinstance(actions, list):
                    # Normalize and validate action format
                    normalized = []
                    validation_errors = []
                    
                    for idx, action in enumerate(actions):
                        if isinstance(action, dict):
                            norm_action = {
                                'name': action.get('name', ''),
                                'arguments': action.get('arguments', {})
                            }
                            
                            # Validate arguments
                            valid, error = self._validate_action_arguments(norm_action)
                            if not valid:
                                validation_errors.append(f"Action {idx} ({norm_action['name']}): {error}")
                                logger.warning(f"Invalid action arguments: {error}")
                                # Try to fix common issues
                                norm_action = self._fix_common_argument_issues(norm_action)
                            
                            normalized.append(norm_action)
                    
                    if validation_errors:
                        logger.info(f"Fixed {len(validation_errors)} argument issues during parsing")
                    
                    return normalized
            
            # Fallback: empty list
            return []
            
        except Exception as e:
            logger.debug(f"Failed to parse actions: {e}")
            return []
    
    def _fix_common_argument_issues(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to fix common argument formatting issues."""
        args = action.get('arguments', {})
        fixed_args = {}
        
        for key, value in args.items():
            # Fix: Convert complex item objects to simple IDs
            if key in ('items_to_return', 'items_to_exchange', 'items'):
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # Extract IDs if present, otherwise use positions
                    if 'item_id' in value[0]:
                        fixed_args['item_ids'] = [item.get('item_id') for item in value]
                    elif 'position' in value[0]:
                        fixed_args['item_positions'] = [item.get('position') for item in value]
                    else:
                        # Use list indices as positions
                        fixed_args['item_positions'] = list(range(1, len(value) + 1))
                    logger.info(f"Fixed complex {key} -> simple item reference")
                    continue
            
            # Fix: Convert nested address objects to strings
            if key in ('new_address', 'shipping_address', 'address'):
                if isinstance(value, dict):
                    # Convert to simple string
                    street = value.get('street', '')
                    city = value.get('city', '')
                    state = value.get('state', '')
                    zip_code = value.get('zip', value.get('zip_code', ''))
                    fixed_args[key] = f"{street}, {city}, {state} {zip_code}".strip(', ')
                    logger.info(f"Fixed nested {key} -> simple string")
                    continue
            
            # Fix: Rename common parameter variations
            if key == 'refund_method' and value == 'original_payment':
                fixed_args['refund_to_original_payment_method'] = True
                logger.info("Fixed refund_method -> refund_to_original_payment_method")
                continue
            
            # Keep as-is
            fixed_args[key] = value
        
        action['arguments'] = fixed_args
        return action
    
    def _generate_outputs(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Generate expected outputs"""
        
        # For most tasks, outputs are empty (actions are the main output)
        # Only generate outputs if query asks for specific information
        
        query_lower = memory.query.lower()
        asks_for_info = any(word in query_lower for word in ['what', 'how much', 'when', 'where', 'status', 'balance'])
        
        if not asks_for_info:
            return {
                'success': True,
                'outputs': [],
                'reasoning': 'Query is action-focused, no info outputs needed'
            }
        
        # Generate output based on actions
        prompt = self._build_outputs_prompt(memory)
        
        try:
            # Use APIClient for retries and robust extraction
            api_client = APIClient(self.client, self.config)
            try:
                content = api_client.call_with_retry(prompt)
            except Exception as e:
                logger.error(f"Outputs generation failed (API): {e}")
                return {'success': False, 'error': str(e), 'outputs': []}

            if not content:
                logger.error("Outputs generation failed: empty response content")
                return {'success': False, 'error': 'Empty content from LLM', 'outputs': []}

            output = content.strip()

            return {
                'success': True,
                'outputs': [output],
                'reasoning': 'Generated informational output for query'
            }
            
        except Exception as e:
            logger.error(f"Outputs generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'outputs': []
            }
    
    def _build_outputs_prompt(self, memory: TaskGenerationMemory) -> str:
        """Build prompt for outputs generation"""
        
        prompt_parts = []
        prompt_parts.append(f"**Query:** {memory.query}")
        prompt_parts.append(f"\n**Actions Taken:**")
        for action in memory.actions:
            prompt_parts.append(f"- {action.get('name', 'unknown')}")
        
        prompt_parts.append("\n**Task:** Generate a brief, helpful response to the customer's query.")
        prompt_parts.append("Output only the response text, no extra formatting.")
        
        return "\n".join(prompt_parts)
    
    def _apply_corrections(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Apply validation corrections"""
        
        if not memory.validation_results:
            return {
                'success': False,
                'error': 'No validation results to correct'
            }
        
        last_validation = memory.validation_results[-1]
        
        # Apply suggestions from validation
        corrections = []
        
        for suggestion in last_validation.suggestions:
            if 'suggest_user_id' in suggestion:
                # Fix user_id in actions
                user_id = suggestion['suggest_user_id']
                for action in memory.actions:
                    if isinstance(action, dict) and 'arguments' in action:
                        args = action['arguments']
                        if isinstance(args, dict) and not args.get('user_id'):
                            args['user_id'] = user_id
                            corrections.append({
                                'action': action.get('name', ''),
                                'fix': f'Added user_id: {user_id}'
                            })
        
        # Fix missing data references
        for missing in last_validation.missing:
            missing_type = missing.get('type', '')
            if missing_type == 'email':
                # Try to use a valid email from prompt_data
                user_data = memory.prompt_data.get('sampled_user_details', {})
                if user_data.get('email'):
                    corrections.append({
                        'fix': f'Using valid email: {user_data["email"]}'
                    })
        
        return {
            'success': True,
            'corrections': corrections,
            'reasoning': f'Applied {len(corrections)} corrections'
        }


class TaskGenerationVerifier:
    """
    Verifier for task generation: Validates generated components
    
    Input: Generated Components, Memory
    Output: Validation Report (v^t), Next Action
    """
    
    def __init__(self, validator: TaskValidator):
        self.validator = validator
    
    def verify(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Verify current task state with OGT-AGT alignment enforcement."""
        
        # Build temporary task for validation
        task = {
            'q': memory.query,
            'agt': memory.actions,
            'ogt': memory.outputs
        }
        
        # Validate using TaskValidator
        validation_report = self.validator.validate(task)
        
        # CRITICAL: Check OGT-AGT consistency if outputs exist
        if memory.outputs:
            try:
                self.validator._validate_q_action_consistency(task, validation_report)
            except Exception as e:
                logger.warning(f"Q-AGT consistency check failed: {e}")
        
        # Determine next action with Q alignment consideration
        if validation_report.valid:
            # Check if OGT has mismatch suggestions
            ogt_mismatch = any(
                s.get('type') == 'q_action_mismatch'
                for s in validation_report.suggestions
            )
            
            if ogt_mismatch:
                next_action = 'add_missing_actions'
                status = 'needs_action_fix'
            elif memory.query and memory.actions:
                next_action = 'complete'
                status = 'success'
            else:
                next_action = 'continue'
                status = 'success'
        elif validation_report.suggestions:
            # Check for Q mismatch in suggestions
            q_mismatch = any(
                s.get('type') == 'q_action_mismatch'
                for s in validation_report.suggestions
            )
            
            if q_mismatch:
                next_action = 'add_missing_actions'
                status = 'needs_action_fix'
            else:
                next_action = 'apply_corrections'
                status = 'needs_correction'
        else:
            next_action = 'retry'
            status = 'failure'
        
        return {
            'validation_report': validation_report,
            'verification_status': status,
            'next_action': next_action,
            'reasoning': self._build_verification_reasoning(validation_report)
        }
    
    def _build_verification_reasoning(self, report: ValidationReport) -> str:
        """Build human-readable verification reasoning"""
        
        if report.valid:
            return "✓ All references validated successfully"
        
        reasons = []
        if report.missing:
            reasons.append(f"Missing {len(report.missing)} data references")
        if report.suggestions:
            reasons.append(f"{len(report.suggestions)} suggestions available")
        
        return " | ".join(reasons) if reasons else "Validation incomplete"


class AgentFlowTaskGenerator:
    """
    Multi-turn iterative task generator following AgentFlow architecture
    
    Architecture: Planner -> Executor -> Verifier -> (Generator if complete)
    
    Turn t:
    1. Planner: Decides what to generate next (query, actions, outputs)
    2. Executor: Generates the component using LLM
    3. Verifier: Validates against real data
    4. Memory: Updated with M^t+1 for next turn
    
    This provides higher-quality tasks through iterative refinement.
    """
    
    def __init__(self,
                 client: OpenAI,
                 config: TauBenchConfig,
                 data_reader: TauBenchDataReader,
                 validator: TaskValidator,
                 max_turns: int = 5,):
        self.planner = TaskGenerationPlanner(client, config)
        self.executor = TaskGenerationExecutor(client, config, data_reader)
        self.verifier = TaskGenerationVerifier(validator)
        self.max_turns = max_turns
        self.config = config
        # API client for optional LLM-generated outputs
        self.api_client = APIClient(client, config)
    
    def generate(self, scenario: str, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a task using multi-turn AgentFlow process
        
        Args:
            scenario: High-level scenario description
            prompt_data: Knowledge base (users, orders, tools, policies)
            
        Returns:
            Generated task with metadata
        """
        
        # Initialize memory
        memory = TaskGenerationMemory(
            turn=1,
            scenario=scenario,
            prompt_data=prompt_data
        )
        
        logger.info(f"Starting AgentFlow task generation for scenario: {scenario[:50]}")
        
        # Multi-turn generation
        for turn in range(1, self.max_turns + 1):
            logger.info(f"=== Generation Turn {turn} ===")
            memory.turn = turn
            
            # Step 1: Plan next component
            logger.info("Planning next component...")
            plan = self.planner.plan(memory)
            logger.info(f"Plan: {plan['reasoning']}")
            
            memory.current_sub_goal = plan['sub_goal']
            
            # Check if complete
            if plan['sub_goal']['component'] == 'complete':
                logger.info("Generation complete!")
                break
            
            # Step 2: Execute generation
            logger.info(f"Generating {plan['sub_goal']['component']}...")
            execution_result = self.executor.execute(plan, memory)
            
            if not execution_result.get('success'):
                # Build diagnostics and let the planner replan with diagnostics
                logger.error(f"Generation failed: {execution_result.get('error')}")
                comp = plan['sub_goal']['component']
                # Increment attempt counter for the component
                memory.component_attempts[comp] = memory.component_attempts.get(comp, 0) + 1

                diagnostics = {
                    'component': comp,
                    'error': execution_result.get('error'),
                    'result': execution_result,
                    'turn': turn,
                    'attempts': memory.component_attempts.get(comp, 0)
                }
                # store diagnostics on memory for audit and planner use
                memory.execution_errors.append(diagnostics)

                # Ask planner for a replan that includes diagnostics
                try:
                    new_plan = self.planner.replan_with_diagnostics(memory, diagnostics)
                    logger.info(f"Planner replan decision: {new_plan.get('reasoning')}")
                except Exception as e:
                    logger.warning(f"Planner replan failed: {e}")
                    new_plan = None

                # If planner returned an immediate replan, try it in the same turn (best-effort)
                if new_plan and turn < self.max_turns:
                    memory.current_sub_goal = new_plan['sub_goal']
                    # execute the replan immediately
                    logger.info(f"Attempting replanned component: {new_plan['sub_goal']['component']}")
                    repl_result = self.executor.execute(new_plan, memory)
                    if repl_result.get('success'):
                        # treat as successful generation and proceed to verification below
                        execution_result = repl_result
                        comp = new_plan['sub_goal']['component']
                    else:
                        # record repl_result diagnostics and advance to next turn
                        memory.execution_errors.append({
                            'component': new_plan['sub_goal']['component'],
                            'error': repl_result.get('error'),
                            'result': repl_result,
                            'turn': turn,
                            'replan': True
                        })
                        if turn < self.max_turns:
                            memory = memory.clone_for_next_turn()
                            continue
                        else:
                            break
                else:
                    # No replan or cannot repro in this turn: go to next turn
                    if turn < self.max_turns:
                        memory = memory.clone_for_next_turn()
                        continue
                    else:
                        break

            # On successful generation, reset attempt counter for that component
            comp = plan['sub_goal']['component']
            if comp in memory.component_attempts:
                memory.component_attempts[comp] = 0
            
            # Update memory with generated component
            component = plan['sub_goal']['component']
            if component == 'query':
                memory.query = execution_result.get('query', '')
                logger.info(f"Query: {memory.query[:100]}")
            elif component == 'actions':
                memory.actions = execution_result.get('actions', [])
                logger.info(f"Actions: {len(memory.actions)} generated")
            elif component == 'outputs':
                memory.outputs = execution_result.get('outputs', [])
                logger.info(f"Outputs: {len(memory.outputs)} generated")
            elif component == 'corrections':
                corrections = execution_result.get('corrections', [])
                memory.corrections_applied.extend(corrections)
                logger.info(f"Corrections: {len(corrections)} applied")
            
            # Step 3: Verify
            logger.info("Verifying...")
            verification = self.verifier.verify(memory)

            # Store both the structured ValidationReport and the full verifier output
            memory.validation_results.append(verification['validation_report'])
            # Append full verification dict so Planner can inspect status/next_action
            memory.verification_history.append(verification)

            # If verifier requests a retry, count that as an attempt for the component just executed
            if verification.get('next_action') == 'retry':
                comp_retry = plan['sub_goal']['component']
                memory.component_attempts[comp_retry] = memory.component_attempts.get(comp_retry, 0) + 1
            
            logger.info(f"Verification: {verification['verification_status']} -> {verification['next_action']}")
            
            # Check verification result
            if verification['next_action'] == 'complete':
                logger.info("Verification passed - task complete!")
                break
            elif verification['next_action'] == 'retry' and turn >= self.max_turns:
                logger.warning("Max turns reached with invalid task")
                break
            
            # Continue to next turn
            if turn < self.max_turns:
                memory = memory.clone_for_next_turn()
        
        # Build final result
        return self._build_result(memory)
    
    def _build_result(self, memory: TaskGenerationMemory) -> Dict[str, Any]:
        """Build final task result from memory - simplified."""
        
        # Get final validation
        validation_report = memory.validation_results[-1] if memory.validation_results else None
        
        task = {
            'q': memory.query,
            'agt': memory.actions,
            'ogt': memory.outputs
        }
        
        # Single validation and correction pass
        validator = self.verifier.validator
        if not validation_report or not validation_report.valid:
            validation_report = validator.validate(task)
            if validation_report.missing or validation_report.suggestions:
                corrections = validator.apply_suggestions(task, validation_report)
                validation_report = validator.validate(task)  # Final check
        
        # Post-process: normalize formats (single pass)
        self._normalize_task_formats(task, validator)
        
        # Generate output if missing
        if not task.get('ogt') and memory.actions:
            try:
                llm_text = self._generate_llm_ogt(memory)
                if llm_text:
                    task['ogt'] = [llm_text]
                else:
                    task['ogt'] = [self._synthesize_ogt_from_actions(task, memory)]
            except Exception:
                task['ogt'] = [self._synthesize_ogt_from_actions(task, memory)]
        
        return {
            'success': validation_report.valid if validation_report else False,
            'task': task,
            'thought': f"Generated via AgentFlow in {memory.turn} turns",
            'raw_response': json.dumps(memory.to_dict(), indent=2),
            'validation_report': {
                'valid': validation_report.valid,
                'missing': validation_report.missing,
                'suggestions': validation_report.suggestions
            } if validation_report else None,
            'corrections_applied': {
                'corrections': memory.corrections_applied
            } if memory.corrections_applied else None,
            'metadata': {
                'generation_method': 'agentflow',
                'turns': memory.turn,
                'scenario': memory.scenario
            }
        }
    
    def _normalize_task_formats(self, task: Dict[str, Any], validator: TaskValidator) -> None:
        """Normalize order IDs and user IDs in single pass."""
        agt = task.get('agt', []) or []
        
        # Normalize order IDs in actions
        for action in agt:
            if not isinstance(action, dict):
                continue
            args = action.get('arguments') or {}
            for key, value in list(args.items()):
                if key.lower() in ('order_id', 'orderid', 'order'):
                    args[key] = validator._normalize_order_id(str(value)) if value else value
                elif key.lower() in ('order_ids', 'orderids'):
                    if isinstance(value, list):
                        args[key] = [validator._normalize_order_id(str(x)) for x in value]
        
        # Enforce single canonical user
        canonical_user = None
        for action in agt:
            if isinstance(action, dict) and action.get('name') == 'find_user_id_by_email':
                canonical_user = action.get('arguments', {}).get('user_id')
                break
        
        if not canonical_user:
            user_ids = [
                a.get('arguments', {}).get('user_id') 
                for a in agt 
                if isinstance(a, dict) and a.get('arguments', {}).get('user_id')
            ]
            if user_ids:
                canonical_user = Counter(user_ids).most_common(1)[0][0]
        
        if canonical_user:
            user_target_actions = {
                'get_user_details', 'get_order_details', 'return_delivered_order_items',
                'exchange_delivered_order_items', 'modify_pending_order_address',
                'modify_pending_order_payment', 'cancel_pending_order', 'modify_pending_order_items',
                'find_user_id_by_email', 'find_user_id_by_name_zip', 'modify_user_address'
            }
            for action in agt:
                if isinstance(action, dict) and action.get('name') in user_target_actions:
                    action.setdefault('arguments', {})['user_id'] = canonical_user
    
    # OGT generation helpers: deterministic synthesis and LLM fallback
    def _synthesize_ogt_from_actions(self,task: Dict[str, Any], memory: Optional['TaskGenerationMemory'] = None) -> str:
        """Deterministically synthesize a friendly agent response from actions.

        Mirrors the behavior previously implemented on AgentFlowTaskGenerator but
        centralized for reuse.
        """
        try:
            actions = task.get('agt', []) or []
            # Extract identifiers
            user_id = None
            email = None
            order_ids: List[str] = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                args = a.get('arguments', {}) or {}
                if not user_id and args.get('user_id'):
                    user_id = args.get('user_id')
                if not email and args.get('email'):
                    email = args.get('email')
                if 'order_id' in args:
                    v = args.get('order_id')
                    if isinstance(v, list):
                        order_ids.extend(v)
                    else:
                        order_ids.append(v)
                if 'order_ids' in args and isinstance(args.get('order_ids'), list):
                    order_ids.extend(args.get('order_ids'))

            action_map = {
                'cancel_pending_order': 'cancelled the pending order',
                'modify_pending_order_address': 'updated the shipping address for the pending order',
                'modify_user_address': 'updated the user address on file',
                'modify_pending_order_payment': 'updated the payment method for the pending order',
                'return_delivered_order_items': 'arranged a return for delivered items',
                'exchange_delivered_order_items': 'processed an exchange for delivered items',
                'modify_pending_order_items': 'modified items on the pending order',
                'get_order_details': 'retrieved order details',
                'get_user_details': 'retrieved user details',
                'find_user_id_by_email': 'located the user account by email',
                'transfer_to_human_agents': 'escalated the request to a human agent'
            }

            performed = []
            for a in actions:
                name = (a.get('name') or '').strip()
                if not name:
                    continue
                desc = action_map.get(name)
                if not desc:
                    desc = name.replace('_', ' ')
                performed.append(desc)

            parts = []
            id_display = user_id or email or 'customer'
            parts.append(f"Hello {id_display},")
            parts.append("Thank you for contacting us. We processed your request as follows:")

            if order_ids:
                unique_orders = sorted(set(str(o) for o in order_ids))
                parts.append(f"- Orders affected: {', '.join(unique_orders)}.")

            if performed:
                for p in performed[:5]:
                    parts.append(f"- {p}.")

            parts.append("If you need further assistance or would like explicit confirmation for any specific change (tracking numbers, refund timing, or payment confirmations), please request it; otherwise, we will proceed with the change and confirm once it is completed.")
            parts.append("Best regards,\nCustomer Support Team")

            return '\n'.join(parts)
        except Exception:
            return "Thank you — your request has been processed."


class APIClient:
    """
    Handles OpenAI API communication with retry logic and response extraction.
    
    Features:
    - Exponential backoff retry strategy
    - Robust response content extraction
    - Configurable timeout and max retries
    """
    
    SYSTEM_PROMPT = (
        "You are an expert at generating realistic retail customer service scenarios. "
        "Generate diverse, realistic scenarios that test different aspects of customer service. "
        "Be concise but thorough."
    )
    
    def __init__(self, client: OpenAI, config: TauBenchConfig):
        """
        Initialize API client.
        
        Args:
            client: OpenAI client instance
            config: Task generation configuration
        """
        self.client = client
        self.config = TauBenchConfig()
    
    def call_with_retry(self, prompt: str) -> str:
        """
        Make API call with exponential backoff retry on failure.
        
        Args:
            prompt: User prompt to send to LLM
            
        Returns:
            Response content string
            
        Raises:
            Exception: After max retries exceeded or on fatal error
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Log prompt size for debugging potential prompt length issues
                try:
                    logger.debug(f"API prompt length: {len(prompt)} characters")
                except Exception:
                    pass

                response = self.client.chat.completions.create(
                    model=self.config.default_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                
                # Extract content with robust error handling
                content = self._extract_content(response)
                if content:
                    return content

                raise ValueError("Empty response from API")
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    backoff = self.config.retry_backoff_base ** attempt
                    logger.warning(
                        f"API call failed (attempt {attempt}/{self.config.max_retries}), "
                        f"retrying in {backoff:.1f}s: {e}"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"API call failed after {self.config.max_retries} attempts: {e}"
                    )
        
        raise last_exception or Exception("API call failed")
    
    @staticmethod
    def _extract_content(response: Any) -> Optional[str]:
        """
        Extract message content from OpenAI API response (robust).
        
        Handles both object and dict response formats.
        
        Args:
            response: API response object or dict
            
        Returns:
            Extracted content string or None
        """
        try:
            # Object attribute access
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                # Common structured attribute layout
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    return choice.message.content
                # Dict-like fallback
                if isinstance(choice, dict):
                    return choice.get('message', {}).get('content')

            # Dict access
            if isinstance(response, dict):
                choices = response.get('choices', [])
                if choices:
                    return choices[0].get('message', {}).get('content')
        except Exception as e:
            logger.debug(f"Failed to extract content using helper: {e}")
            return None


class TauBenchOpenAIGenerator:
    """
    Complete TauBench task generator using OpenAI API and real data.
    
    This is the main orchestrator that combines:
    - Data reading (TauBenchDataReader)
    - API communication (APIClient)
    - Response parsing (ResponseParser)
    - Task validation (TaskValidator)
    
    Supports two generation modes:
    1. Direct generation: Single LLM call with full prompt
    2. AgentFlow generation: Multi-turn iterative refinement with tool execution
    """
    
    def __init__(
        self, 
        envs_path: str = "envs/retail",
        use_agentflow: bool = False,
        agentflow_max_turns: int = 5
    ):
        """
        Initialize the task generator.
        
        Args:
            envs_path: Path to environment data directory
            use_agentflow: Enable AgentFlow multi-turn generation
            agentflow_max_turns: Maximum turns for AgentFlow generation
        """
        self.data_reader = TauBenchDataReader(envs_path)
        self.config = TauBenchConfig()
        self.use_agentflow = use_agentflow
        
        # Initialize API client
        api_key = os.environ.get('OPENAI_API_KEY', self.config.default_api_key)
        base_url = os.environ.get('OPENAI_BASE_URL', self.config.default_base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Initialize components
        self.api_client = APIClient(self.client, self.config)
        self.validator = TaskValidator(self.data_reader)
        
        # Initialize AgentFlow generator if requested
        self.agentflow_generator = None
        if use_agentflow:
            self.agentflow_generator = AgentFlowTaskGenerator(
                client=self.client,
                config=self.config,
                data_reader=self.data_reader,
                validator=self.validator,
                max_turns=agentflow_max_turns,
            )
            logger.info(f"Initialized with AgentFlow mode (max {agentflow_max_turns} turns)")
        
        # Cache for tokenizer
        self._tokenizer: Optional[Any] = None
        self._normalization_cache: Dict[str, Any] = {}  # 新增：规范化缓存
    
    @lru_cache(maxsize=1)
    def _get_tokenizer(self) -> Any:
        """Get or create tiktoken encoder (cached)."""
        try:
            return tiktoken.encoding_for_model(self.config.default_model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text.split())

    def _detect_scenario_from_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Detect scenario key from task using config.scenario_action_map.

        Returns the best matching scenario key or None if none match.
        """
        if not isinstance(task, dict):
            return None
        agt = task.get('agt', []) or []
        if not isinstance(agt, list) or not agt:
            return None

        action_names = [a.get('name') for a in agt if isinstance(a, dict) and a.get('name')]
        if not action_names:
            return None

        # Score scenarios by number of matching actions
        scores = {}
        for key, actions in self.config.scenario_action_map.items():
            if not isinstance(actions, (list, tuple, set)):
                continue
            score = sum(1 for a in action_names if a in actions)
            if score:
                scores[key] = score

        if not scores:
            return None
        return max(scores.items(), key=lambda x: x[1])[0]

    def _normalize_task_format(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize task format for consistency across generation methods.
        
        Enhanced with caching and optimized processing.
        """
        if not isinstance(task, dict):
            return {'q': '', 'agt': [], 'ogt': []}

        # Use cached result if available
        task_hash = hash(json.dumps(task, sort_keys=True))
        if task_hash in self._normalization_cache:
            return self._normalization_cache[task_hash]

        cleaned = self._normalize_task_structure(task)
        cleaned = self._normalize_actions(cleaned)
        cleaned = self._normalize_order_references(cleaned)
        cleaned = self._normalize_outputs(cleaned)
        cleaned = self._enforce_user_consistency(cleaned)
        
        # Cache the result
        self._normalization_cache[task_hash] = cleaned
        return cleaned
    
    def _normalize_task_structure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize basic task structure."""
        cleaned = dict(task)  # shallow copy
        
        # Normalize instruction
        if not cleaned.get('q'):
            cleaned['q'] = (
                cleaned.get('query') or 
                cleaned.get('instruction') or 
                ''
            )
        
        return cleaned
    
    def _normalize_actions(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize actions with deduplication and validation."""
        agt = task.get('agt')
        if agt is None:
            # Try alternative keys
            agt = (
                task.get('actions') or 
                task.get('expected_actions') or 
                []
            )

        # Normalize and deduplicate actions
        normalized_actions = self._process_actions_list(agt)
        task['agt'] = normalized_actions
        
        # Validate required fields
        validation_issues = self._validate_required_fields(normalized_actions)
        if validation_issues:
            task['_validation_issues'] = validation_issues
        
        return task
    
    def _process_actions_list(self, actions: Any) -> List[Dict[str, Any]]:
        """Process and deduplicate actions list."""
        if not isinstance(actions, list):
            return []
        
        normalized_actions = []
        seen_signatures = set()
        
        for action in actions:
            normalized = self._normalize_single_action(action)
            if normalized:
                signature = self._get_action_signature(normalized)
                if signature not in seen_signatures:
                    normalized_actions.append(normalized)
                    seen_signatures.add(signature)
        
        return normalized_actions
    
    def _normalize_single_action(self, action: Any) -> Optional[Dict[str, Any]]:
        """Normalize a single action."""
        if isinstance(action, dict):
            name = action.get('name') or action.get('function') or action.get('action') or ''
            args = self._normalize_arguments(action.get('arguments') or action.get('args') or action.get('parameters') or {})
            return {'name': name, 'arguments': args}
        elif isinstance(action, str):
            return {'name': action, 'arguments': {}}
        return None
    
    def _normalize_arguments(self, args: Any) -> Dict[str, Any]:
        """Normalize action arguments."""
        if not isinstance(args, dict):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return args or {}
    
    def _get_action_signature(self, action: Dict[str, Any]) -> Tuple[str, str]:
        """Create signature for action deduplication."""
        try:
            name = action.get('name', '')
            args = action.get('arguments', {})
            args_str = json.dumps(args, sort_keys=True)
            return (name, args_str)
        except Exception:
            return (action.get('name', ''), str(action.get('arguments', {})))
    
    def _validate_required_fields(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate required fields for actions."""
        required_map = {
            'cancel_pending_order': ['order_id'],
            'modify_pending_order_address': ['order_id'],
            'modify_user_address': ['user_id'],
            'modify_pending_order_payment': ['order_id'],
            'return_delivered_order_items': [['order_id', 'order_ids']],
            'exchange_delivered_order_items': ['order_id'],
            'modify_pending_order_items': ['order_id'],
            'get_order_details': ['order_id'],
            'get_user_details': ['user_id'],
            'find_user_id_by_email': ['email']
        }
        
        validation_issues = []
        
        for idx, action in enumerate(actions):
            name = (action.get('name') or '').strip()
            args = action.get('arguments') or {}
            
            if name in required_map:
                missing = self._check_required_fields(args, required_map[name])
                if missing:
                    validation_issues.append({
                        'action_index': idx, 
                        'action': name, 
                        'missing': missing
                    })
        
        return validation_issues
    
    def _check_required_fields(self, args: Dict[str, Any], requirements: Any) -> List[str]:
        """Check presence of required fields given requirement spec.

        Supports alternative requirements like [['order_id','order_ids']].
        """
        if not isinstance(requirements, (list, tuple)) or not requirements:
            return []

        # Alternative requirements expressed as a list within the first element
        if isinstance(requirements[0], list):
            alternatives = requirements[0]
            for key in alternatives:
                if args.get(key):
                    return []
            return alternatives
        else:
            missing: List[str] = []
            for key in requirements:
                if not args.get(key):
                    missing.append(key)
            return missing
    
    def _normalize_order_references(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all order references in the task."""
        # Normalize actions
        for action in task.get('agt', []):
            if isinstance(action, dict):
                action['arguments'] = self._normalize_order_ids_in_dict(action.get('arguments', {}))
        
        # Normalize top-level fields
        order_fields = ['order_id', 'order_ids', 'order']
        for field in order_fields:
            if field in task:
                task[field] = self._normalize_order_id_value(task[field])
        
        return task
    
    def _normalize_order_ids_in_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize order IDs in a dictionary."""
        normalized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower in ('order_id', 'orderid', 'order', 'order_ids', 'orderids'):
                normalized[key] = self._normalize_order_id_value(value)
            else:
                normalized[key] = value
        return normalized
    
    def _normalize_order_id_value(self, value: Any) -> Any:
        """Normalize a single order ID value."""
        if value is None:
            return value
        
        if isinstance(value, (str, int)):
            try:
                return _normalize_order_id_global(str(value))
            except Exception:
                return ('#' + str(value).lstrip('#'))
        elif isinstance(value, list):
            return [self._normalize_order_id_value(item) for item in value]
        else:
            return value
    
    def _normalize_outputs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize outputs field."""
        ogt = task.get('ogt')
        if ogt is None:
            ogt = task.get('outputs') or task.get('expected_outputs') or []
        task['ogt'] = ogt if isinstance(ogt, list) else [ogt]
        return task
    
    def _enforce_user_consistency(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce single user_id consistency across all actions."""
        try:
            agt = task.get('agt', []) or []
            canonical_user = self._find_canonical_user_id(agt)
            
            if canonical_user:
                user_target_actions = {
                    'get_user_details', 'get_order_details', 'return_delivered_order_items',
                    'exchange_delivered_order_items', 'modify_pending_order_address',
                    'modify_pending_order_payment', 'cancel_pending_order', 'modify_pending_order_items',
                    'find_user_id_by_email', 'find_user_id_by_name_zip', 'modify_user_address'
                }
                
                for action in agt:
                    if isinstance(action, dict):
                        name = action.get('name')
                        if name in user_target_actions:
                            args = action.setdefault('arguments', {})
                            if args.get('user_id') != canonical_user:
                                args['user_id'] = canonical_user
        
        except Exception:
            # Be permissive on errors
            pass
        
        return task
    
    def _find_canonical_user_id(self, actions: List[Dict[str, Any]]) -> Optional[str]:
        """Determine canonical user_id from actions (prefer find_user_id_by_email then most common)."""
        if not isinstance(actions, list):
            return None
        for action in actions:
            if isinstance(action, dict) and action.get('name') == 'find_user_id_by_email':
                args = action.get('arguments') or {}
                if args.get('user_id'):
                    return args.get('user_id')

        user_ids = [
            action.get('arguments', {}).get('user_id')
            for action in actions
            if isinstance(action, dict) and action.get('arguments', {}).get('user_id')
        ]
        user_ids = [uid for uid in user_ids if uid]
        if user_ids:
            return Counter(user_ids).most_common(1)[0][0]
        return None
    
    def _validate_and_correct_task(
        self,
        task: Dict[str, Any],
        log_prefix: str = "Task"
    ) -> Tuple[Dict[str, Any], ValidationReport, Optional[Dict[str, Any]]]:
        """
        Consolidated validation and correction logic.
        """
        if not isinstance(task, dict):
            task = {}
        
        # Normalize and validate
        task = self._normalize_task_format(task)
        validation_report = self.validator.validate(task)
        corrections = None
        
        # Apply corrections if needed
        if validation_report and (validation_report.missing or validation_report.suggestions):
            logger.info(f"Applying corrections to {log_prefix}...")
            corrections = self.validator.apply_suggestions(task, validation_report)
            
            # Re-normalize and re-validate after corrections
            task = self._normalize_task_format(task)
            validation_report = self.validator.validate(task)
        
        return task, validation_report, corrections
    
    def _build_result_dict(
        self,
        task: Dict[str, Any],
        validation_report: Optional[ValidationReport],
        corrections: Optional[Dict[str, Any]],
        thought: str = "",
        raw_response: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build standardized result dictionary."""
        result = {
            "thought": thought,
            "task": task,
            "raw_response": raw_response,
            "success": validation_report.valid if validation_report else True,
            "validation_report": {
                'valid': validation_report.valid,
                'missing': validation_report.missing,
                'suggestions': validation_report.suggestions
            } if validation_report else None,
            "corrections_applied": corrections
        }
        
        if metadata:
            result["metadata"] = metadata
        
        return result
    
    def generate_task_with_real_data(
        self,
        custom_user_id: Optional[str] = None,
        include_metadata: bool = False,
        force_mode: Optional[str] = None,
        suggested_scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a task using real data.
        """
        use_agentflow_mode = force_mode == 'agentflow' or (force_mode != 'direct' and self.use_agentflow)
        retries = getattr(self.config, 'scenario_match_retries', 2)
        strict = getattr(self.config, 'scenario_match_strict', True)

        last_result = None
        for attempt in range(1, retries + 1):
            if use_agentflow_mode:
                last_result = self._generate_with_agentflow(custom_user_id, include_metadata, suggested_scenario)
            else:
                last_result = self._generate_with_direct_mode(custom_user_id, include_metadata, suggested_scenario)

            # If generation produced a task, detect scenario and compare
            task = last_result.get('task') if isinstance(last_result, dict) else None
            detected = self._detect_scenario_from_task(task) if task else None
            # Ensure metadata exists
            last_result.setdefault('metadata', {})
            last_result['metadata']['suggested_scenario'] = suggested_scenario
            last_result['metadata']['detected_scenario'] = detected
            last_result['metadata']['suggested_scenario_matched'] = (suggested_scenario == detected)

            if not suggested_scenario or suggested_scenario == detected:
                # Good: we either didn't ask for a scenario or it matches
                return last_result
            # If not matched and strict and we have attempts left, try again (log)
            if strict and attempt < retries:
                logger.info(f"Suggested scenario '{suggested_scenario}' did not match detected '{detected}' - retrying generation (attempt {attempt+1}/{retries})")
                continue
            # fallback: return last result, with mismatch metadata
            if not strict:
                logger.warning(f"Suggested scenario '{suggested_scenario}' did not match detected '{detected}' - returning anyway due to non-strict mode")
            else:
                logger.warning(f"Failed to generate a task matching suggested scenario '{suggested_scenario}' after {retries} attempts; returning last result")
            return last_result
    
    def _generate_with_agentflow(
        self,
        custom_user_id: Optional[str],
        include_metadata: bool,
        suggested_scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate task using AgentFlow mode."""
        if not self.agentflow_generator:
            return {
                "error": "AgentFlow mode not enabled",
                "success": False
            }
        
        try:
            prompt_data = self._prepare_prompt_data(custom_user_id)
            # Respect suggested_scenario (if provided) when generating the scenario
            scenario = self._generate_scenario(prompt_data, suggested_scenario)
            logger.info(f"Generated scenario: {scenario}")
            
            result = self.agentflow_generator.generate(scenario, prompt_data)
            
            if include_metadata:
                result.setdefault("metadata", {}).update({
                    "custom_user_id": custom_user_id,
                    "model": self.config.default_model,
                    "generation_method": "agentflow",
                    "scenario": scenario,
                    "suggested_scenario": suggested_scenario
                })
            
            return result
            
        except Exception as e:
            logger.error(f"AgentFlow generation failed: {e}")
            return {
                "error": f"AgentFlow generation failed: {str(e)}",
                "success": False
            }
    
    def _generate_with_direct_mode(
        self,
        custom_user_id: Optional[str],
        include_metadata: bool,
        suggested_scenario: Optional[str]
    ) -> Dict[str, Any]:
        """Generate task using Direct mode."""
        try:
            prompt_data = self._prepare_prompt_data(custom_user_id)
            
            # Generate blueprint
            blueprint_prompt = self._build_blueprint_prompt(prompt_data, suggested_scenario)
            blueprint_raw = self.api_client.call_with_retry(blueprint_prompt)
            blueprint_parse = self._committee_parse(blueprint_raw)
            
            # Validate and correct blueprint
            blueprint_task = blueprint_parse.task or {}
            blueprint_task.setdefault('ogt', [])
            
            blueprint_task, validation_report, corrections = self._validate_and_correct_task(
                blueprint_task, "blueprint"
            )
            
            # Instantiate if valid
            if validation_report and validation_report.valid:
                result = self._instantiate_blueprint(blueprint_task, prompt_data)
            else:
                result = self._build_result_dict(
                    task=blueprint_task,
                    validation_report=validation_report,
                    corrections=corrections,
                    thought="Blueprint generation completed with validation issues",
                    raw_response=blueprint_parse.raw_response
                )
            
            # Add metadata
            if include_metadata:
                token_info = self._calculate_token_info(prompt_data)
                result.setdefault("metadata", {}).update({
                    "custom_user_id": custom_user_id,
                    "model": self.config.default_model,
                    "temperature": self.config.temperature,
                    "prompt_token_info": token_info,
                    "suggested_scenario": suggested_scenario
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            return {
                "error": f"Task generation failed: {str(e)}",
                "success": False
            }
    
    def _generate_scenario(self, prompt_data: Dict[str, Any], suggested_scenario: Optional[str] = None) -> str:
        """Generate a scenario from available data."""
        scenario_templates = list(self.config.scenario_templates.values())
        
        # If a suggested scenario is provided, try to map it to a template and return
        if suggested_scenario and suggested_scenario in self.config.scenario_templates:
            return self.config.scenario_templates[suggested_scenario]

        # Filter scenarios based on order status using centralized action map
        orders = prompt_data.get('sampled_orders', [])
        if orders and isinstance(orders[0], dict):
            order = orders[0]
            status = order.get('status', '').lower()
            # Work with keys, then map to templates
            keys = list(self.config.scenario_keys)
            if status == 'pending':
                # Keep scenarios that have pending-specific actions
                pending_actions = {'cancel_pending_order', 'modify_pending_order_items', 'modify_pending_order_address', 'modify_pending_order_payment'}
                keys = [k for k in keys if any(a in pending_actions for a in self.config.scenario_action_map.get(k, []))]
            elif status == 'delivered':
                delivered_actions = {'return_delivered_order_items', 'exchange_delivered_order_items'}
                keys = [k for k in keys if any(a in delivered_actions for a in self.config.scenario_action_map.get(k, []))]
            scenario_templates = [self.config.scenario_templates[k] for k in keys]
        
        return random.choice(scenario_templates)
    
    def _prepare_prompt_data(self, custom_user_id: Optional[str]) -> Dict[str, Any]:
        """Prepare prompt data with user-order relationship awareness."""
        prompt_data = self.data_reader.generate_complete_prompt_data()
        
        try:
            data_files = self.data_reader.read_data_files()
            users = data_files.get('users', {})
            orders = data_files.get('orders', {})
            products = data_files.get('products', {})
            
            # Build user-order mapping
            user_order_map = defaultdict(list)
            for order_id, order in orders.items():
                if isinstance(order, dict) and (user_id := order.get('user_id')):
                    user_order_map[user_id].append(order)
            
            # Select target user
            target_user = self._select_target_user(custom_user_id, users, user_order_map)
            if target_user:
                user_id = target_user.get('user_id') or target_user.get('id')
                prompt_data["sampled_user_details"] = target_user
                
                # Select user's orders
                prompt_data["sampled_orders"] = self._select_user_orders(user_id, user_order_map, orders)
                # Enrich orders with product details where available so prompts always include real product data
                try:
                    sampled_orders = prompt_data.get("sampled_orders") or []
                    enriched_products = {}
                    for order in sampled_orders:
                        if not isinstance(order, dict):
                            continue
                        items = order.get('items') or []
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            # product id can be under different keys in various formats
                            pid = item.get('product_id') or item.get('pid') or item.get('product') or item.get('sku')
                            if not pid:
                                continue
                            pid_str = str(pid)
                            # look up product details and attach
                            if pid_str in products:
                                prod = products.get(pid_str)
                                # store product details in a local map and attach a lightweight view to item
                                enriched_products[pid_str] = prod
                                item['product_details'] = {
                                    'product_id': pid_str,
                                    'name': (prod.get('name') or prod.get('title')) if isinstance(prod, dict) else prod,
                                    'variant_id': prod.get('variant_id') if isinstance(prod, dict) else None,
                                    'sku': prod.get('sku') if isinstance(prod, dict) else None
                                }
                    # Also include sampled_products as a convenience mapping for prompts
                    if enriched_products:
                        prompt_data['sampled_products'] = enriched_products
                except Exception as e:
                    logger.debug(f"Failed to enrich sampled orders with products: {e}")
                        
        except Exception as e:
            logger.warning(f"Error in data preparation: {e}")
        
        return prompt_data
    
    def _select_target_user(self, custom_user_id: Optional[str], users: Dict, user_order_map: Dict) -> Optional[Dict]:
        """Select target user for generation."""
        if custom_user_id and custom_user_id in users:
            return self.data_reader.sample_user_data(custom_user_id)
        
        # Select user with orders
        users_with_orders = [uid for uid in users.keys() if user_order_map.get(uid)]
        if users_with_orders:
            selected_uid = random.choice(users_with_orders)
            return self.data_reader.sample_user_data(selected_uid)
        
        # Fallback to any user
        if users:
            return self.data_reader.sample_user_data(next(iter(users.keys())))
        
        return None
    
    def _select_user_orders(self, user_id: Optional[str], user_order_map: Dict, all_orders: Dict) -> List[Dict]:
        """Select orders for the target user."""
        if user_id and user_id in user_order_map:
            user_orders = user_order_map[user_id]
            num_orders = min(len(user_orders), random.randint(1, 3))
            # deep-copy to avoid mutating original data
            return [copy.deepcopy(o) for o in random.sample(user_orders, num_orders)]
        else:
            # Fallback to random orders
            all_orders_list = list(all_orders.values())
            if all_orders_list:
                num_orders = min(len(all_orders_list), random.randint(1, 3))
                orders = random.sample(all_orders_list, num_orders)
                logger.warning(f"User {user_id} has no orders, using random orders")
                return [copy.deepcopy(o) for o in orders]
        
        return []
    
    def _calculate_token_info(self, prompt_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Calculate token counts for prompt pieces."""
        pieces = {
            "task_rules": prompt_data["task_rules"],
            "domain_rules": prompt_data["domain_rules"],
            "sampled_user_details": json.dumps(prompt_data["sampled_user_details"], separators=(",", ":")),
            "sampled_orders": json.dumps(prompt_data["sampled_orders"], separators=(",", ":")),
            "sampled_personas": json.dumps(prompt_data.get("sampled_personas", []), separators=(",", ":")),
            "sampled_policy": json.dumps(prompt_data.get("sampled_policy", {}), separators=(",", ":")),
            "sampled_api": json.dumps(prompt_data.get("sampled_api", {}), separators=(",", ":")),
            "sampled_tools": prompt_data.get("sampled_tools", "[]"),
            "sampled_domain_data": json.dumps(prompt_data.get("sampled_domain_data", {}), separators=(",", ":")),
            "example": prompt_data["example"]
        }
        
        return {
            name: {"token_count": self._count_tokens(text)}
            for name, text in pieces.items()
        }

    # 保留 _committee_parse 和 _build_blueprint_prompt 方法（它们没有重复逻辑）
    def _committee_parse(self, response_content: str, committee_size: int = 3) -> ParseResult:
        """Attempt parsing with a small committee of lightweight strategies."""
        # 原有实现保持不变
        variants = []
        variants.append(response_content)
        variants.append(response_content.strip())
        try:
            no_code = re.sub(r"```.*?```", "", response_content, flags=re.S)
            variants.append(no_code)
        except Exception:
            variants.append(response_content)
        variants.append(re.sub(r"\s+", " ", response_content).strip())

        tried = []
        last_error = None
        seen = set()
        count = 0
        
        for v in variants:
            if count >= committee_size:
                break
            if v in seen:
                continue
            seen.add(v)
            count += 1

            try:
                pr = ResponseParser.parse(v)
            except Exception as e:
                pr = ParseResult(success=False, task=None, thought="", error=str(e), raw_response=v)

            # Fallback: accept JSON without thought
            if not pr.success:
                try:
                    # Use centralized JSON extraction helper for fallback parsing
                    answer_only = _extract_json_candidate_from_text(v)
                    if isinstance(answer_only, dict) and 'q' in answer_only and 'agt' in answer_only:
                        pr = ParseResult(success=True, task=answer_only, thought="", raw_response=v)
                except Exception:
                    pass

            pr.candidates_tried = tried + [v]
            tried.append(v)

            if pr.success:
                logger.debug(f"Committee parse succeeded on variant #{count}")
                return pr
            last_error = pr.error or last_error

        return ParseResult(
            success=False, 
            task=None, 
            thought="", 
            error=last_error or "parse_failed", 
            raw_response=response_content, 
            candidates_tried=tried
        )

    def _build_blueprint_prompt(self, prompt_data: Dict[str, Any], suggested_scenario: Optional[str] = None) -> str:
            """Build a compact prompt that requests only a blueprint: q and agt (actions).

            The blueprint should be output as JSON containing keys 'q' and 'agt' where
            'agt' is a list of actions with 'name' and 'arguments'. Keep the response
            concise and machine-parseable.
            
            Args:
                prompt_data: Data dictionary with user/order/tool information
                suggested_scenario: Optional scenario to prioritize (e.g., 'order_cancellation', 'item_return')
            """
            parts = []
            parts.append("## Instructions")
            parts.append("Generate a compact blueprint for a single customer service task.")
            parts.append("The blueprint should contain:")
            parts.append("  - 'q': A realistic user query that MUST include authentication information:")
            parts.append("    * If using find_user_id_by_email: MUST mention the user's EMAIL in the query")
            parts.append("    * If using find_user_id_by_name_zip: MUST mention the user's FULL NAME and ZIP CODE in the query")
            parts.append("    * If using find_user_id_by_username: MUST mention the USERNAME in the query")
            parts.append("  - 'agt': A list of actions (tool calls) with 'name' and 'arguments'")
            parts.append("")
            parts.append("⚠️  CRITICAL: The query ('q') MUST contain the exact authentication data needed by the FIRST action!")
            parts.append("   Examples:")
            parts.append("   - If first action is find_user_id_by_email → query MUST include: 'My email is john@example.com'")
            parts.append("   - If first action is find_user_id_by_name_zip → query MUST include: 'My name is John Doe and my zip code is 12345'")
            parts.append("")
            
            # Add scenario forcing if provided
            if suggested_scenario:
                parts.append(f"🎯 REQUIRED SCENARIO TYPE: {suggested_scenario.replace('_', ' ').title()}")
                parts.append(f"You MUST generate a task for this specific scenario type.")
                parts.append("")
            
            parts.append("CRITICAL: You MUST use the actual data references provided below. Do not invent user IDs, order IDs, product IDs, product names, or other data - use only what is provided in the User Data and Order Data sections.")
            parts.append("If you mention a product in the 'q' or 'agt', you MUST use the EXACT product name and product_id from the Order Data above.")
            
            # Show full user details as JSON
            user_details = prompt_data.get('sampled_user_details', {})
            parts.append("\n### User Data (USE THESE EXACT USER IDs AND DETAILS)")
            parts.append(json.dumps(user_details, indent=2))
            
            # Show full order details as JSON
            orders = prompt_data.get('sampled_orders', [])
            if orders:
                parts.append("\n### Order Data (USE THESE EXACT ORDER IDs AND DETAILS)")
                parts.append(json.dumps(orders, indent=2))
            
            # Include policy snippets if available
            sampled_policy = prompt_data.get('sampled_policy', {})
            if sampled_policy:
                parts.append("\n### Policy Snippets (important constraints to consider)")
                parts.append(json.dumps(sampled_policy, indent=2))
            
            # Include persona if available
            sampled_personas = prompt_data.get('sampled_personas', [])
            if sampled_personas:
                parts.append("\n### Suggested Persona(s)")
                parts.append(json.dumps(sampled_personas, indent=2))
                parts.append("Use this persona to shape the user's tone and communication style in 'q'.")
            
            # Guidelines for actions
            parts.append("\n## Guidelines for generating Actions (agt)")
            parts.append("1. Provide precise tool calls with all necessary parameters for each action.")
            parts.append("2. Each action MUST include the correct 'user_id' from the User Data above.")
            parts.append("3. ALL actions in the same task MUST use the SAME user_id - never mix different users.")
            parts.append("4. Use actual order_id values from the Order Data above.")
            parts.append("5. Arguments should be fully populated - NO empty objects {}.")
            parts.append("6. Vary action sequences - not all tasks need the same pattern of tools.")
            parts.append("7. Consider the user's actual order status and items when designing actions.")
            parts.append("8. IMPORTANT: Do NOT generate actions that are incompatible with the order's status. "
                         "If an order is 'delivered', DO NOT use modify_pending_* or cancel_pending_order — "
                         "use return_delivered_order_items, exchange_delivered_order_items, or transfer_to_human_agents. "
                         "If your agent response (ogt) promises an update (address change, expedited shipping, refund), "
                         "you MUST include the corresponding action in 'agt' that performs that change.")
            parts.append("9. ADDRESS FORMAT: For address updates (modify_pending_order_address or modify_user_address), the action 'arguments' MUST include the structured fields: 'address1', 'address2' (may be empty), 'city', 'state', 'country', and 'zip'. Do NOT use a single free-form 'new_address' string in place of structured fields.")
            parts.append("10. PAYMENT & REFUND: For refunds, returns, or exchanges, include an explicit 'payment_method_id' in the action arguments. Use the actual 'payment_method_id' from the User Data when available; otherwise use the placeholder value 'ORIGINAL_PAYMENT_METHOD'.")
            
            # Include tools
            parts.append("\n## Tools")
            parts.append("The available tool combination in Python format is as follows:")
            parts.append(prompt_data.get('sampled_tools', '[]'))
            
            # Add scenario suggestions
            parts.append("\n### Scenario Suggestions (pick ONE or create your own):")
            parts.append("  - Order Cancellation: cancel_pending_order")
            parts.append("  - Address Change: modify_pending_order_address or modify_user_address")
            parts.append("  - Item Return: return_delivered_order_items")
            parts.append("  - Item Exchange: exchange_delivered_order_items")
            parts.append("  - Order Modification: modify_pending_order_items, modify_pending_order_payment")
            parts.append("  - Account Inquiry: get_user_details, get_order_details")
            parts.append("  - Product Information: get_product_details, list_all_product_types")

            parts.append("\n## Output Format")
            parts.append("Generate your response in strict JSON format, without any additional comments or explanations.")
            parts.append("Return a single JSON object with keys 'q' and 'agt'.")
            parts.append("Do NOT include 'ogt' or any extra commentary.")
            parts.append("Do NOT wrap in markdown code fences.")

            # Add concrete example
            parts.append("\n## Example Task")
            parts.append("Here is an example of the expected output format with concrete values.")
            parts.append("⚠️  Do NOT directly copy instruction and action patterns from the example - create unique scenarios!")
            parts.append("")
            parts.append('{')
            parts.append('  "q": "Hi, this is Ethan Lopez. I just moved and need to update my default shipping address to 742 Evergreen Terrace, Apt 5B, Springfield, IL, USA, 62704. Also, could you tell me how much balance I have left on my gift card?",')
            parts.append('  "agt": [')
            parts.append('    {')
            parts.append('      "name": "find_user_id_by_email",')
            parts.append('      "arguments": {')
            parts.append('        "email": "ethan.lopez8943@example.com",')
            parts.append('        "user_id": "ethan_lopez_6291"')
            parts.append('      }')
            parts.append('    },')
            parts.append('    {')
            parts.append('      "name": "get_user_details",')
            parts.append('      "arguments": {')
            parts.append('        "user_id": "ethan_lopez_6291"')
            parts.append('      }')
            parts.append('    },')
            parts.append('    {')
            parts.append('      "name": "modify_user_address",')
            parts.append('      "arguments": {')
            parts.append('        "user_id": "ethan_lopez_6291",')
            parts.append('        "address1": "742 Evergreen Terrace",')
            parts.append('        "address2": "Apt 5B",')
            parts.append('        "city": "Springfield",')
            parts.append('        "state": "IL",')
            parts.append('        "country": "USA",')
            parts.append('        "zip": "62704"')
            parts.append('      }')
            parts.append('    }'),
            parts.append('    {'),
            parts.append('      "name": "exchange_delivered_order_items",'),
            parts.append('      "arguments": {'),
            parts.append('        "order_id": "#W0000000",'),
            parts.append('        "item_ids": ["4273929280"],'),
            parts.append('        "new_item_ids": ["4273929280"],'),
            parts.append('        "payment_method_id": "gift_card_0000000"'),
            parts.append('      }'),
            parts.append('    }')
            parts.append('  ]')
            parts.append('}')
            parts.append("")
            parts.append("Notice: ALL three actions use the same user_id 'ethan_lopez_6291' consistently.")
            parts.append("This is the pattern you MUST follow - use ONE user_id across ALL actions.")
            parts.append("")
            parts.append("Now generate a DIFFERENT scenario using the data provided above. Be creative with:")
            parts.append("  - Different user personality/tone than the example")
            parts.append("  - Different scenario type (cancellation, return, inquiry, etc.)")
            parts.append("  - Different combination of tools")
            parts.append("  - Different level of detail/urgency")
            parts.append("")
            parts.append("Generate the unique blueprint now.")
            
            return "\n".join(parts)

    def _instantiate_blueprint(self, blueprint_task: Dict[str, Any], prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate a validated blueprint by generating output."""
        try:
            task = {
                'q': blueprint_task.get('q', ''),
                'agt': blueprint_task.get('agt', []),
                'ogt': []
            }
            
            # Generate output
            ogt = self._generate_output_for_blueprint(blueprint_task, prompt_data)
            task['ogt'] = [ogt] if ogt else []
            
            # Validate final task
            task, validation_report, corrections = self._validate_and_correct_task(
                task, "Instantiated blueprint"
            )
            
            return self._build_result_dict(
                task=task,
                validation_report=validation_report,
                corrections=corrections,
                thought="Blueprint instantiated with generated output",
                raw_response=json.dumps(blueprint_task, indent=2)
            )
            
        except Exception as e:
            logger.error(f"Blueprint instantiation failed: {e}")
            return {"error": str(e), "success": False}
    
    def _generate_output_for_blueprint(self, blueprint_task: Dict[str, Any], prompt_data: Dict[str, Any]) -> str:
        """Generate agent response for blueprint."""
        query = blueprint_task.get('q', '')
        actions = blueprint_task.get('agt', [])
        # Prefer LLM-based generation via centralized helper
        try:
            return _generate_llm_ogt(TaskGenerationMemory(turn=0, scenario="", prompt_data=prompt_data, query=query, actions=actions, outputs=[]), self.api_client) or "Thank you for contacting us. Your request has been processed successfully."
        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            return "Thank you for contacting us. Your request has been processed successfully."
    
    
    def generate_diverse_tasks(self, 
                              num_tasks: int = 5,
                              progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """Generate multiple diverse tasks"""
        tasks = []
        
        # Get available user IDs
        try:
            data = self.data_reader.read_data_files()
            user_ids = list(data.get('users', {}).keys())
            random.shuffle(user_ids)
        except Exception as e:
            logger.warning(f"Failed to get user IDs: {e}")
            user_ids = []
        
        for i in range(num_tasks):
            logger.info(f"Generating task {i+1}/{num_tasks}...")
            
            if progress_callback:
                progress_callback(i + 1, num_tasks)
            
            user_id = user_ids[i % len(user_ids)] if user_ids else None
            
            # Choose generation method
            task = self.generate_task_with_real_data(custom_user_id=user_id)
            
            tasks.append(task)
            # Log failures only
            if not task.get('success'):
                logger.error(f"Task {i+1} failed: {task.get('error')}")
        
        return tasks
    
    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], filename: str = "generated_tasks.json"):
        """Save generated tasks to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"Tasks saved to {filename}")
    
    def display_task_summary(self, task: Dict[str, Any]):
        """Display a summary of a generated task"""
        # Handle wrapped result or direct task
        if isinstance(task, dict) and 'success' in task and 'task' in task:
            if not task.get("success"):
                print(f"❌ Task failed: {task.get('error')}")
                return
            task_data = task.get("task")
        else:
            task_data = task
        
        # Handle list-wrapped tasks
        if isinstance(task_data, list):
            task_data = next((x for x in task_data if isinstance(x, dict)), None)
            if not task_data:
                print("(Task is a list without dict elements)")
                return
        
        print("📋 Task Summary:")
        print("-" * 40)
        
        # Display instruction
        instruction = task_data.get("q", "")
        if len(instruction) > self.config.max_instruction_display_length:
            instruction = instruction[:self.config.max_instruction_display_length] + "..."
        print(f"Instruction: {instruction}")
        
        # Display actions
        actions = task_data.get("agt", [])
        print(f"Actions: {len(actions)} tool calls")
        
        if actions:
            action_types = [action.get("name", "unknown") for action in actions]
            unique_actions = list(set(action_types))
            print(f"Tool types: {', '.join(unique_actions)}")
        
        # Display outputs
        outputs = task_data.get("ogt", [])
        if outputs:
            print(f"Outputs: {len(outputs)} information responses")


def main():
    """Main function demonstrating the complete generator with both modes"""
    
    print("🚀 Tau-Bench Task Generator with AgentFlow Support")
    print("=" * 60)
    
    # Ask user which mode to use
    print("\nGeneration Modes:")
    print("1. Direct Mode (faster, blueprint-based)")
    print("2. AgentFlow Mode (higher quality, multi-turn iterative)")
    print("3. Both (compare modes)")
    
    choice = input("\nSelect mode (1/2/3, default 1): ").strip() or "1"
    
    use_agentflow = choice in ["2", "3"]
    use_direct = choice in ["1", "3"]
    
    # Initialize generator(s)
    try:
        # Always create direct mode generator
        direct_generator = TauBenchOpenAIGenerator("envs/retail", use_agentflow=False)
        logger.info("Direct mode generator initialized")
        
        # Create AgentFlow generator if requested
        agentflow_generator = None
        if use_agentflow:
            agentflow_generator = TauBenchOpenAIGenerator("envs/retail", use_agentflow=True, agentflow_max_turns=5)
            logger.info("AgentFlow generator initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        print("Make sure 'envs/retail' directory exists with the required files")
        return
    
    # Generate single task
    print(f"\n{'='*60}")
    print("🎯 Generating tasks...")
    
    # Direct mode generation
    if use_direct:
        print(f"\n{'='*60}")
        print("📦 Direct Mode Generation (blueprint-based)")
        print("=" * 60)
        
        result_direct = direct_generator.generate_task_with_real_data()
        
        if result_direct.get("success"):
            print("✅ Direct mode task generated successfully!")
            
            # Display thought process
            if result_direct.get("thought"):
                print(f"\n💭 Generation Process:")
                print("-" * 30)
                print(result_direct["thought"])
            
            # Display validation report
            if result_direct.get("validation_report"):
                report = result_direct["validation_report"]
                print(f"\n🔍 Validation Report:")
                print(f"  Valid: {'✅' if report.get('valid') else '❌'}")
                
                if report.get('missing'):
                    print(f"  Missing references: {len(report['missing'])}")
                
                if result_direct.get("corrections_applied"):
                    corrections = result_direct["corrections_applied"].get("corrections", [])
                    if corrections:
                        print(f"  ✨ Corrections applied: {len(corrections)}")
            
            # Display task summary
            print(f"\n📊 Task Details:")
            direct_generator.display_task_summary(result_direct)
            
            # Display full task
            print(f"\n📋 Complete Generated Task (Direct Mode):")
            print(json.dumps(result_direct["task"], indent=2))
            
        else:
            print(f"❌ Direct mode generation failed: {result_direct.get('error')}")
    
    # AgentFlow mode generation
    if use_agentflow and agentflow_generator:
        print(f"\n{'='*60}")
        print("🔄 AgentFlow Mode Generation (multi-turn iterative)")
        print("=" * 60)
        
        result_agentflow = agentflow_generator.generate_task_with_real_data()
        
        if result_agentflow.get("success"):
            print("✅ AgentFlow task generated successfully!")
            
            # Display metadata
            metadata = result_agentflow.get("metadata", {})
            if metadata:
                print(f"\n📊 Generation Metadata:")
                print(f"  Method: {metadata.get('generation_method', 'unknown')}")
                print(f"  Turns: {metadata.get('turns', 'unknown')}")
                if metadata.get('scenario'):
                    print(f"  Scenario: {metadata['scenario']}")
            
            # Display thought process
            if result_agentflow.get("thought"):
                print(f"\n💭 Generation Process:")
                print("-" * 30)
                print(result_agentflow["thought"])
            
            # Display validation report
            if result_agentflow.get("validation_report"):
                report = result_agentflow["validation_report"]
                print(f"\n🔍 Validation Report:")
                print(f"  Valid: {'✅' if report.get('valid') else '❌'}")
            
            # Display task summary
            print(f"\n📊 Task Details:")
            agentflow_generator.display_task_summary(result_agentflow)
            
            # Display full task
            print(f"\n📋 Complete Generated Task (AgentFlow Mode):")
            print(json.dumps(result_agentflow["task"], indent=2))
            
        else:
            print(f"❌ AgentFlow generation failed: {result_agentflow.get('error')}")
    
    # Compare results if both modes were used
    if use_direct and use_agentflow and result_direct.get("success") and result_agentflow.get("success"):
        print(f"\n{'='*60}")
        print("📊 Mode Comparison")
        print("=" * 60)
        
        direct_task = result_direct["task"]
        agentflow_task = result_agentflow["task"]
        
        print(f"\nDirect Mode:")
        print(f"  Query length: {len(direct_task.get('q', ''))} chars")
        print(f"  Actions: {len(direct_task.get('agt', []))}")
        print(f"  Outputs: {len(direct_task.get('ogt', []))}")
        
        print(f"\nAgentFlow Mode:")
        print(f"  Query length: {len(agentflow_task.get('q', ''))} chars")
        print(f"  Actions: {len(agentflow_task.get('agt', []))}")
        print(f"  Outputs: {len(agentflow_task.get('ogt', []))}")
    
    # Ask for multiple tasks
    print(f"\n{'='*60}")
    generate_multiple = input("Generate multiple diverse tasks? (y/n): ").lower() == 'y'
    
    if generate_multiple:
        num_tasks = int(input("How many tasks? (default 3): ") or "3")
        
        # Choose which generator to use
        generator = agentflow_generator if (use_agentflow and not use_direct) else direct_generator
        mode_name = "AgentFlow" if (use_agentflow and not use_direct) else "Direct"
        
        print(f"\n🔄 Generating {num_tasks} diverse tasks using {mode_name} mode...")
        
        # Progress callback
        def show_progress(current: int, total: int):
            print(f"Progress: {current}/{total} tasks completed")
        
        tasks = generator.generate_diverse_tasks(
            num_tasks, 
            progress_callback=show_progress
        )
        
        # Summary
        successful_tasks = [t for t in tasks if t.get("success")]
        print(f"\n📊 Generation Summary:")
        print(f"✅ Successful: {len(successful_tasks)}/{len(tasks)}")
        
        # Display summaries
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Task {i} ---")
            generator.display_task_summary(task)
        
        # Save to file
        save_file = input(f"\nSave all tasks to file? (y/n): ").lower() == 'y'
        if save_file:
            filename = input("Filename (default: generated_tasks.json): ").strip() or "generated_tasks.json"
            generator.save_tasks_to_file(tasks, filename)
    
    print(f"\n🎉 Task generation completed!")


if __name__ == "__main__":
    main()