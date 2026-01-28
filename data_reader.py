import json
import os
import re
import importlib.util
import importlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import random
import datetime
from configs import TauBenchConfig

# Setup logger
logger = logging.getLogger(__name__)


class TauBenchDataReader:
    """Reads and processes tau-bench retail domain data from source files"""
    
    def __init__(self, envs_path: str = "envs/retail", sampling_config: Optional[dict] = None, seed: Optional[int] = None):
        """
        Initialize the data reader
        
        Args:
            envs_path: Path to the retail environment directory
        """
        self.envs_path = Path(envs_path)
        self.data_cache = {}
        self.seed = seed
        self.configs = TauBenchConfig()
        if seed is not None:
            random.seed(seed)

        # Sampling configuration: expects dict with (min,max) ranges for samplers
        self.sampling_config = sampling_config or {}
        # Preload all data into cache for fast access
        try:
            self.load_all_data()
        except Exception:
            # If preload fails (e.g., running from a different cwd), defer until needed
            pass
    
    def read_wiki_policy(self) -> Dict[str, str]:
        """Read the retail agent policy from wiki.md"""
        wiki_path = self.envs_path / "wiki.md"
        
        if not wiki_path.exists():
            raise FileNotFoundError(f"Wiki file not found: {wiki_path}")
        
        with open(wiki_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract different sections from the wiki
        sections = {}
        
        # Extract main policy
        policy_match = re.search(r'# Retail agent policy\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if policy_match:
            sections['main_policy'] = policy_match.group(1).strip()
        
        # Extract domain basics
        domain_match = re.search(r'## Domain basic\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if domain_match:
            sections['domain_basic'] = domain_match.group(1).strip()
        
        # Extract specific operation rules
        cancel_match = re.search(r'## Cancel pending order\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if cancel_match:
            sections['cancel_rules'] = cancel_match.group(1).strip()
        
        modify_match = re.search(r'## Modify pending order\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if modify_match:
            sections['modify_rules'] = modify_match.group(1).strip()
        
        return_match = re.search(r'## Return delivered order\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if return_match:
            sections['return_rules'] = return_match.group(1).strip()
        
        exchange_match = re.search(r'## Exchange delivered order\n\n(.*?)(?=##|$)', content, re.DOTALL)
        if exchange_match:
            sections['exchange_rules'] = exchange_match.group(1).strip()
        
        return sections
    
    def read_data_files(self) -> Dict[str, Any]:
        """Read user, product, and order data from JSON files"""
        data_path = self.envs_path / "data"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        data = {}
        
        # Read users.json
        users_path = data_path / "users.json"
        if users_path.exists():
            with open(users_path, 'r', encoding='utf-8') as f:
                data['users'] = json.load(f)
        
        # Read products.json
        products_path = data_path / "products.json"
        if products_path.exists():
            with open(products_path, 'r', encoding='utf-8') as f:
                data['products'] = json.load(f)
        
        # Read orders.json
        orders_path = data_path / "orders.json"
        if orders_path.exists():
            with open(orders_path, 'r', encoding='utf-8') as f:
                data['orders'] = json.load(f)
        
        # cache for later fast lookup
        self.data_cache['users'] = data.get('users', {})
        self.data_cache['products'] = data.get('products', {})
        self.data_cache['orders'] = data.get('orders', {})

        return data
    
    def read_tools_info(self) -> List[Dict[str, Any]]:
        """Read tool information from the tools directory using actual tool classes"""
        try:
            # Import the tools package to get actual tool classes
            tools_module_name = "envs.retail.tools"
            tools_pkg = importlib.import_module(tools_module_name)
            
            # Get all tools
            all_tools = getattr(tools_pkg, "ALL_TOOLS", None)
            if not all_tools:
                raise ImportError("No ALL_TOOLS found in envs.retail.tools")
            
            tools = []
            for tool_cls in all_tools:
                try:
                    if hasattr(tool_cls, 'get_info'):
                        info = tool_cls.get_info()
                        # Ensure the info is in OpenAI function calling format
                        if 'function' in info and 'name' in info['function']:
                            tools.append(info)
                except Exception as e:
                    logger.debug(f"Failed to get info for tool {tool_cls}: {e}")
            
            return tools
            
        except Exception as e:
            logger.warning(f"Failed to load real tools, falling back to file parsing: {e}")
            # Fallback to the original file-based parsing
            return self._read_tools_info_fallback()

    def _read_tools_info_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to read tool information from files when import fails"""
        tools_path = self.envs_path / "tools"
        
        if not tools_path.exists():
            raise FileNotFoundError(f"Tools directory not found: {tools_path}")
        
        tools = []
        
        # Read __init__.py to get tool list
        init_path = tools_path / "__init__.py"
        if init_path.exists():
            with open(init_path, 'r', encoding='utf-8') as f:
                init_content = f.read()
            
            # Extract ALL_TOOLS list
            all_tools_match = re.search(r'ALL_TOOLS = \[(.*?)\]', init_content, re.DOTALL)
            if all_tools_match:
                tool_names = re.findall(r'(\w+),?', all_tools_match.group(1))
                
                for tool_name in tool_names:
                    tool_info = self._extract_tool_info(tools_path, tool_name)
                    if tool_info:
                        tools.append(tool_info)
        
        return tools
    
    def _extract_tool_info(self, tools_path: Path, tool_name: str) -> Optional[Dict[str, Any]]:
        """Extract tool information in OpenAI function calling format"""
        # Convert CamelCase to snake_case for filename
        filename = re.sub(r'(?<!^)(?=[A-Z])', '_', tool_name).lower() + '.py'
        tool_path = tools_path / filename
        
        if not tool_path.exists():
            return None
        
        try:
            with open(tool_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to extract function info using get_info method pattern
            get_info_match = re.search(r'def get_info\(\) -> Dict\[str, Any\]:(.*?)return\s*({.*?})', content, re.DOTALL)
            if get_info_match:
                try:
                    return_dict_str = get_info_match.group(2)
                    
                    # Extract function name
                    name_match = re.search(r'"name":\s*"([^"]*)"', return_dict_str)
                    function_name = name_match.group(1) if name_match else tool_name.lower()
                    
                    # Extract description
                    desc_match = re.search(r'"description":\s*\(\s*"([^"]*)"', return_dict_str)
                    if not desc_match:
                        desc_match = re.search(r'"description":\s*"([^"]*)"', return_dict_str)
                    description = desc_match.group(1) if desc_match else f"Tool for {function_name}"
                    
                    # Try to extract parameters from the invoke method
                    parameters = self._extract_tool_parameters(content)
                    
                    # Return in OpenAI function calling format
                    return {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": parameters,
                                "required": list(parameters.keys())
                            }
                        }
                    }
                    
                except Exception as e:
                    logger.debug(f"Failed to parse tool {tool_name}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.debug(f"Error reading tool {tool_name}: {e}")
            return None
    
    def _extract_tool_parameters(self, content: str) -> Dict[str, Any]:
        """Extract tool parameters from the invoke method signature"""
        parameters = {}
        
        # Look for invoke method signature
        invoke_match = re.search(r'def invoke\(self,\s*data:\s*Dict\[str,\s*Any\](?:,\s*(.*?))?\)\s*->', content, re.DOTALL)
        if invoke_match and invoke_match.group(1):
            params_str = invoke_match.group(1).strip()
            if params_str:
                # Parse parameters
                for param in params_str.split(','):
                    param = param.strip()
                    if ':' in param:
                        param_name = param.split(':')[0].strip()
                        param_type = param.split(':')[1].strip()
                        
                        # Convert Python types to JSON schema types
                        json_type = "string"  # default
                        if "int" in param_type.lower():
                            json_type = "integer"
                        elif "float" in param_type.lower():
                            json_type = "number"
                        elif "bool" in param_type.lower():
                            json_type = "boolean"
                        elif "list" in param_type.lower() or "List" in param_type:
                            json_type = "array"
                        elif "dict" in param_type.lower() or "Dict" in param_type:
                            json_type = "object"
                        
                        parameters[param_name] = {
                            "type": json_type,
                            "description": f"Parameter {param_name} of type {param_type}"
                        }
        
        return parameters

    def read_personas(self, persona_file: str = None) -> List[str]:
        """Read personas from a persona.jsonl file (one JSON per line)

        Returns list of persona strings.
        """
        path = Path(persona_file) if persona_file else Path("persona.jsonl")
        if not path.exists():
            return []

        personas = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and 'persona' in obj:
                            personas.append(obj['persona'])
                        elif isinstance(obj, str):
                            personas.append(obj)
                    except Exception:
                        # Fallback: treat the raw line as persona
                        personas.append(line)
        except Exception:
            return []

        return personas

    def sample_persona(self, num: int = 1) -> List[str]:
        """Sample one or more personas for task generation"""
        personas = self.read_personas()
        if not personas:
            return []
        if num <= 0:
            return []
        # If requesting more than available, return all shuffled
        if num >= len(personas):
            shuffled = personas.copy()
            random.shuffle(shuffled)
            return shuffled
        return random.sample(personas, min(num, len(personas)))

    def api_sampler(self, num_write: int = 7, num_read: int = 9) -> Dict[str, Any]:
        """Sample APIs (tools) and prioritize 'write' APIs for core actions while allowing flexibility for 'read' APIs."""
        tools = self.read_tools_info()
        write_tools = []
        read_tools = []

        for t in tools:
            name = t.get('name', '') or t.get('function_name', '') or t.get('filename', '')
            desc = (t.get('description') or '').lower()
            lname = str(name).lower()

            is_write = any(kw in lname or kw in desc for kw in self.configs.write_keywords)

            if is_write:
                write_tools.append(t)
            else:
                read_tools.append(t)

        # Helper function to sample tools
        def _sample_list(lst, n):
            if not lst:
                return []
            if n >= len(lst):
                random.shuffle(lst)
                return lst
            return random.sample(lst, n)

        # Sample 'write' APIs as the core
        sampled_write = _sample_list(write_tools, num_write)

        # Allow flexibility in sampling 'read' APIs
        sampled_read = _sample_list(read_tools, num_read)

        return {
            'all_tools': tools,
            'write_tools': sampled_write,
            'read_tools': sampled_read,
        }


    def policy_sampler(self, num_sections: int = 2, rules_per_section: int = 3) -> Dict[str, Any]:
        """Sample from the wiki policy sections. Returns short lists of rule snippets with metadata."""
        policy = self.read_wiki_policy()
        if not policy:
            return {}

        sections = list(policy.keys())
        if not sections:
            return {}
        if num_sections >= len(sections):
            sampled_sections = sections.copy()
            random.shuffle(sampled_sections)
        else:
            sampled_sections = random.sample(sections, min(num_sections, len(sections)))
        sampled = {}
        for sec in sampled_sections:
            text = policy.get(sec, '')
            # Split into sentences and pick a few
            sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', text) if s.strip()]
            picks = random.sample(sentences, min(rules_per_section, len(sentences))) if sentences else []
            sampled[sec] = {
                'source': sec,
                'rules': picks
            }

        return sampled

    def domain_data_sampler(self, num_products: int = 3, num_orders: int = 2, include_metadata: bool = True) -> Dict[str, Any]:
        """Sample products and orders and enrich them with lightweight metadata (cost, est_time, attributes)."""
        # Use full cached data and sample from it for diversity
        all_products = self.data_cache.get('products') or self.read_data_files().get('products', {})
        all_orders = self.data_cache.get('orders') or self.read_data_files().get('orders', {})

        products = []
        orders = []

        prod_items = list(all_products.items())
        order_items = list(all_orders.items())

        if prod_items:
            if num_products >= len(prod_items):
                random.shuffle(prod_items)
                products = [p for _, p in prod_items]
            else:
                sampled = random.sample(prod_items, num_products)
                products = [p for _, p in sampled]

        if order_items:
            if num_orders >= len(order_items):
                random.shuffle(order_items)
                orders = [o for _, o in order_items]
            else:
                sampled = random.sample(order_items, num_orders)
                orders = [o for _, o in sampled]

        def enrich_product(p):
            enriched = p.copy()
            # Derive a representative price from variants if present
            price = None
            variants = enriched.get('variants') or {}
            if isinstance(variants, dict) and variants:
                # pick first variant price
                for v in variants.values():
                    if isinstance(v, dict) and 'price' in v:
                        price = v.get('price')
                        break
            if not price:
                price = enriched.get('price') or random.uniform(10, 500)

            enriched['representative_price'] = price
            if include_metadata:
                enriched['estimated_shipping_days'] = random.randint(1, 7)
                enriched['fragile'] = any(k in (enriched.get('name') or '').lower() for k in ['glass', 'camera', 'perfume', 'monitor', 'laptop'])
                enriched['popularity'] = random.choice(['low', 'medium', 'high'])
            return enriched

        def enrich_order(o):
            enriched = o.copy()
            total = 0.0
            for it in enriched.get('items', []) or []:
                total += float(it.get('price', 0))
            enriched['order_total_estimate'] = total
            if include_metadata:
                # simple heuristic: pending -> longer expected handling time
                status = enriched.get('status', '').lower()
                if status == 'pending':
                    enriched['expected_resolution_time_hours'] = random.randint(6, 72)
                elif status == 'processed':
                    enriched['expected_resolution_time_hours'] = random.randint(1, 24)
                else:
                    enriched['expected_resolution_time_hours'] = random.randint(0, 6)
            return enriched

        enriched_products = [enrich_product(p) for p in products]
        enriched_orders = [enrich_order(o) for o in orders]

        return {
            'products': enriched_products,
            'orders': enriched_orders
        }

    def example_sampler(self, num_examples: int = 1) -> List[Dict[str, Any]]:
        """Sample few-shot example tasks from the domain tasks.

        Returns a list of example task dicts (q/agt/ogt) when available.
        """
        examples = self.read_example_tasks()
        if not examples:
            # try to parse the single example string
            ex = self.get_example_task()
            try:
                ex_obj = json.loads(ex)
                return [ex_obj]
            except Exception:
                return []

        # examples is expected to be a list of dicts
        if isinstance(examples, list):
            if num_examples >= len(examples):
                out = examples.copy()
                random.shuffle(out)
                return out
            return random.sample(examples, min(num_examples, len(examples)))
        return []

    def sample_frequencies(self) -> Dict[str, int]:
        """Randomize sampling frequency for different samplers for diversity per iteration."""
        # Use sampling_config ranges if provided
        cfg = self.sampling_config or {}
        def _range_pick(key, default_min, default_max):
            v = cfg.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return random.randint(int(v[0]), int(v[1]))
            return random.randint(default_min, default_max)

        freqs = {
            'num_personas': _range_pick('num_personas', 1, 2),
            'num_write_apis': _range_pick('num_write_apis', 1, 7),
            'num_read_apis': _range_pick('num_read_apis', 0, 6),
            'num_policy_sections': _range_pick('num_policy_sections', 1, 6),
            'rules_per_section': _range_pick('rules_per_section', 1, 6),
            'num_products': _range_pick('num_products', 1, 5),
            'num_orders': _range_pick('num_orders', 0, 3),
            'num_examples': _range_pick('num_examples', 0, 2)
        }
        return freqs

    def load_all_data(self):
        """Load and cache all relevant domain files: users, products, orders, tools, policy, examples, personas."""
        # Force reading and caching all data
        try:
            self.read_data_files()
        except Exception:
            pass

        try:
            self.data_cache['policy'] = self.read_wiki_policy()
        except Exception:
            self.data_cache['policy'] = {}

        try:
            self.data_cache['tools'] = self.read_tools_info()
        except Exception:
            self.data_cache['tools'] = []

        try:
            self.data_cache['examples'] = self.read_example_tasks()
        except Exception:
            self.data_cache['examples'] = []

        try:
            self.data_cache['personas'] = self.read_personas()
        except Exception:
            self.data_cache['personas'] = []

        return self.data_cache
    
    def read_example_tasks(self) -> List[Dict[str, Any]]:
        """Read example tasks from tasks.py"""
        tasks_path = self.envs_path / "tasks.py"
        
        # Return cached if already loaded
        if 'examples' in self.data_cache:
            return self.data_cache['examples']

        if not tasks_path.exists():
            self.data_cache['examples'] = []
            return []

        try:
            # Import the tasks module directly
            spec = importlib.util.spec_from_file_location("tasks_module", tasks_path)
            tasks_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tasks_module)
            
            # Get the tasks list
            if hasattr(tasks_module, 'tasks'):
                tasks_list = tasks_module.tasks
                self.data_cache['examples'] = tasks_list
                return tasks_list
            else:
                print("Warning: No 'tasks' variable found in tasks.py")
                self.data_cache['examples'] = []
                return []
                
        except Exception as e:
            print(f"Warning: Could not import tasks module: {e}")
            # Fallback to text parsing method
            parsed = self._parse_tasks_from_text(tasks_path)
            self.data_cache['examples'] = parsed
            return parsed
    
    def _parse_tasks_from_text(self, tasks_path: Path) -> List[Dict[str, Any]]:
        """Fallback method to parse tasks from text"""
        try:
            with open(tasks_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the tasks list and try to evaluate it safely
            tasks_match = re.search(r'tasks = (\[.*?\])', content, re.DOTALL)
            if tasks_match:
                tasks_str = tasks_match.group(1)
                # Use ast.literal_eval for safety, but it might not handle complex structures
                try:
                    import ast
                    tasks = ast.literal_eval(tasks_str)
                    return tasks
                except (ValueError, SyntaxError):
                    print("Warning: Could not safely parse tasks with ast.literal_eval")
                    return []
            
            return []
            
        except Exception as e:
            print(f"Warning: Fallback parsing failed: {e}")
            return []
    
    def sample_user_data(self, user_id: str = None) -> Dict[str, Any]:
        """Sample user data for task generation"""
        data = self.read_data_files()
        users = data.get('users', {})

        if user_id and user_id in users:
            uid = user_id
        elif users:
            # Pick a random user id for diversity
            uid = random.choice(list(users.keys()))
        else:
            return {}

        user = users.get(uid, {})
        
        # Normalize user data structure
        if isinstance(user, dict):
            # Handle different name formats
            if 'name' in user and isinstance(user['name'], dict):
                # Convert {"first_name": "John", "last_name": "Doe"} to "John Doe"
                name_dict = user['name']
                full_name = f"{name_dict.get('first_name', '')} {name_dict.get('last_name', '')}".strip()
                user['name'] = full_name
            
            # Ensure user_id is present
            user['user_id'] = user.get('user_id', uid)
        
        return user
    
    def sample_order_data(self, num_orders: int = 3) -> List[Dict[str, Any]]:
        """Sample order data for task generation"""
        data = self.read_data_files()
        orders = data.get('orders', {})
        
        if orders:
            # Return first few orders as samples and normalize structure
            order_list = []
            for order_id, order_data in list(orders.items())[:num_orders]:
                # Ensure order has order_id field
                if isinstance(order_data, dict):
                    normalized_order = order_data.copy()
                    if 'order_id' not in normalized_order:
                        normalized_order['order_id'] = order_id
                    order_list.append(normalized_order)
            return order_list
        else:
            return []
    
    def sample_product_data(self, num_products: int = 3) -> List[Dict[str, Any]]:
        """Sample product data for task generation"""
        data = self.read_data_files()
        products = data.get('products', {})
        
        if products:
            # Return first few products as samples and normalize structure
            product_list = []
            for product_id, product_data in list(products.items())[:num_products]:
                if isinstance(product_data, dict):
                    normalized_product = product_data.copy()
                    if 'product_id' not in normalized_product:
                        normalized_product['product_id'] = product_id
                    product_list.append(normalized_product)
            return product_list
        else:
            return []
    
    def generate_task_rules(self) -> str:
        """Generate task rules from policy and examples"""
        policy = self.read_wiki_policy()
        
        rules = """
1. Generate realistic customer service scenarios requiring user authentication first
2. Include varied customer personalities (polite, urgent, confused, private, rude, etc.)
3. Focus on common retail operations: cancel, modify, return, exchange orders
4. Include profile management: address updates, payment method changes
5. Make requests specific with actual order numbers, product details, user information
6. Ensure authentication via email or name+zip code before any database actions
7. Include scenarios requiring explicit user confirmation for consequential actions
8. Consider policy constraints (pending vs delivered orders, payment method limitations)
"""
        
        if 'main_policy' in policy:
            rules += f"\n\nBased on retail policy:\n{policy['main_policy']}"
        
        return rules
    
    def generate_domain_rules(self) -> str:
        """Generate domain-specific rules from policy"""
        policy = self.read_wiki_policy()
        
        domain_rules = "RETAIL DOMAIN RULES:\n"
        
        if 'domain_basic' in policy:
            domain_rules += f"BASICS:\n{policy['domain_basic']}\n\n"
        
        if 'cancel_rules' in policy:
            domain_rules += f"CANCEL RULES:\n{policy['cancel_rules']}\n\n"
        
        if 'modify_rules' in policy:
            domain_rules += f"MODIFY RULES:\n{policy['modify_rules']}\n\n"
        
        if 'return_rules' in policy:
            domain_rules += f"RETURN RULES:\n{policy['return_rules']}\n\n"
        
        if 'exchange_rules' in policy:
            domain_rules += f"EXCHANGE RULES:\n{policy['exchange_rules']}\n\n"
        
        return domain_rules
    
    def format_tools_for_prompt(self) -> str:
        """Format tools information for the prompt"""
        tools = self.read_tools_info()
        
        # Tools are already in OpenAI format, just return them as JSON
        return json.dumps(tools, indent=2)
    
    def get_example_task(self) -> str:
        """Get a formatted example task"""
        tasks = self.read_example_tasks()
        
        if tasks:
            # Return first task as example, formatted properly
            task = tasks[0]
            example = {
                "q": task.get("instruction", ""),
                "agt": task.get("actions", []),
                "ogt": task.get("outputs", [])
            }
            return json.dumps(example, indent=2)
        else:
            # Provide a fallback example if tasks.py parsing fails
            fallback_example = {
                "q": "Hi, I'm Sarah Johnson and I live in Springfield, IL 62701. I need to cancel my recent order because I ordered it by mistake. The order number is #W2024001.",
                "agt": [
                    {
                        "name": "find_user_id_by_name_zip",
                        "arguments": {"first_name": "Sarah", "last_name": "Johnson", "zip": "62701"}
                    },
                    {
                        "name": "get_order_details", 
                        "arguments": {"order_id": "#W2024001"}
                    },
                    {
                        "name": "cancel_pending_order",
                        "arguments": {"order_id": "#W2024001", "reason": "ordered by mistake"}
                    }
                ],
                "ogt": []
            }
            return json.dumps(fallback_example, indent=2)
    
    def generate_complete_prompt_data(self) -> Dict[str, Any]:
        """Generate all data needed for the OpenAI prompt"""
        # Base data
        base_user = self.sample_user_data()
        base_orders = self.sample_order_data()
        base_products = self.sample_product_data()

        # Decide frequencies for this generation iteration
        freqs = self.sample_frequencies()

        # Persona sampling
        sampled_personas = self.sample_persona(freqs.get('num_personas', 1))

        # API sampling (write vs read)
        api_sample = self.api_sampler(num_write=freqs.get('num_api_write', 7),
                                      num_read=freqs.get('num_api_read', 6))

        # Policy sampler
        sampled_policy = self.policy_sampler(num_sections=6,
                                             rules_per_section=freqs.get('rules_per_section', 2))

        # Domain data sampler
        sampled_domain = self.domain_data_sampler(num_products=freqs.get('num_products', 3),
                                                  num_orders=freqs.get('num_orders', 2))

        # Example sampler
        sampled_examples = self.example_sampler(num_examples=freqs.get('num_examples', 2))

        prompt_data = {
            "task_rules": self.generate_task_rules(),
            "domain_rules": self.generate_domain_rules(),
            "sampled_user_details": base_user,
            "sampled_orders": base_orders,
            "sampled_products": base_products,
            "sampled_tools": self.format_tools_for_prompt(),
            "example": self.get_example_task(),

            # Extended samplers for richer prompts
            "sampled_personas": sampled_personas,
            "sampled_api": api_sample,
            "sampled_policy": sampled_policy,
            "sampled_domain_data": sampled_domain,
            "sampled_examples": sampled_examples,
        }

        return prompt_data


def main():
    """Demo the data reader functionality"""
    
    # Initialize reader (adjust path as needed)
    reader = TauBenchDataReader("envs/retail")
    
    print("🔍 Reading tau-bench retail data...")
    print("=" * 50)
    
    try:
        # Read policy
        print("📋 Reading wiki policy...")
        policy = reader.read_wiki_policy()
        print(f"Found {len(policy)} policy sections")
        
        # Read data files
        print("💾 Reading data files...")
        data = reader.read_data_files()
        print(f"Users: {len(data.get('users', {}))}")
        print(f"Products: {len(data.get('products', {}))}")
        print(f"Orders: {len(data.get('orders', {}))}")
        
        # Read tools
        print("🛠️  Reading tools...")
        tools = reader.read_tools_info()
        print(f"Found {len(tools)} tools")
        
        # Read example tasks
        print("📝 Reading example tasks...")
        tasks = reader.read_example_tasks()
        print(f"Found {len(tasks)} example tasks")
        
        print("\n" + "=" * 50)
        print("✅ Successfully read all data!")
        
        # Generate complete prompt data
        print("\n🎯 Generating complete prompt data...")
        prompt_data = reader.generate_complete_prompt_data()
        
        print(f"Generated data keys: {list(prompt_data.keys())}")
        
        # Optional: Save to file
        save_data = input("\nSave data to file? (y/n): ").lower() == 'y'
        if save_data:
            output_file = "tau_bench_prompt_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Data saved to {output_file}")
        
        # Display sample
        print(f"\n📊 Sample user data:")
        sample_user = prompt_data["sampled_user_details"]
        if isinstance(sample_user, dict):
            print(f"User ID: {sample_user.get('user_id', 'N/A')}")
            print(f"Name: {sample_user.get('name', 'N/A')}")
            print(f"Email: {sample_user.get('email', 'N/A')}")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you're running from the correct directory with access to envs/retail/")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

@dataclass
class SamplingConfig:
    num_personas: Tuple[int, int] = (1, 2)
    num_write_apis: Tuple[int, int] = (1, 7)
    num_read_apis: Tuple[int, int] = (0, 6)
    num_policy_sections: Tuple[int, int] = (1, 6)
    rules_per_section: Tuple[int, int] = (1, 6)
    num_products: Tuple[int, int] = (1, 5)
    num_orders: Tuple[int, int] = (0, 3)
    num_examples: Tuple[int, int] = (0, 2)