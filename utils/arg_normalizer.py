from typing import Dict, Any, List
from task_generator import _normalize_order_id_global


def normalize_action_arguments(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common argument mismatches to match tool invoke signatures.

    Rules implemented:
      - modify_pending_order_address / modify_user_address: accept 'new_address' -> expand to address fields
      - cancel_pending_order: ensure 'reason' exists (default 'no longer needed')
      - return_delivered_order_items / exchange_delivered_order_items: map 'refund_to_original_payment_method' to 'payment_method_id' placeholder; ensure 'new_item_ids' exists for exchange
      - get_order_details: normalize order_id
    """
    if not isinstance(args, dict):
        return args or {}

    out = dict(args)  # copy

    # Normalize order_id common formats
    if 'order_id' in out and out['order_id']:
        out['order_id'] = _normalize_order_id_global(out['order_id'])

    # Address expansion
    if name in ('modify_pending_order_address', 'modify_user_address'):
        if 'new_address' in out and out['new_address']:
            na = out.pop('new_address')
            # Simple expansion: put full string in address1 and empty else
            out.setdefault('address1', na)
            out.setdefault('address2', '')
            out.setdefault('city', '')
            out.setdefault('state', '')
            out.setdefault('country', '')
            out.setdefault('zip', '')
        else:
            # Ensure required fields exist
            out.setdefault('address1', out.get('address1', ''))
            out.setdefault('address2', out.get('address2', ''))
            out.setdefault('city', out.get('city', ''))
            out.setdefault('state', out.get('state', ''))
            out.setdefault('country', out.get('country', ''))
            out.setdefault('zip', out.get('zip', ''))

    # Cancel: ensure reason
    if name == 'cancel_pending_order':
        if 'reason' not in out or not out.get('reason'):
            # Heuristic: if refund flag present, choose a default reason
            if out.get('refund_to_original_payment_method'):
                out['reason'] = 'no longer needed'
            else:
                out['reason'] = 'no longer needed'

    # Return / Exchange: payment method and new_item_ids
    if name in ('return_delivered_order_items', 'exchange_delivered_order_items'):
        # Map refund flag to payment_method_id placeholder
        if out.get('refund_to_original_payment_method') and 'payment_method_id' not in out:
            out['payment_method_id'] = 'ORIGINAL_PAYMENT_METHOD'

        # For exchange, ensure new_item_ids exists
        if name == 'exchange_delivered_order_items':
            if 'new_item_ids' not in out or not out.get('new_item_ids'):
                # Fallback: copy item_ids if present
                item_ids = out.get('item_ids') or []
                out['new_item_ids'] = list(item_ids)
            # Ensure payment_method_id exists (use placeholder if missing)
            out.setdefault('payment_method_id', 'ORIGINAL_PAYMENT_METHOD')

        # For returns, ensure payment_method_id exists when refund flag present
        if name == 'return_delivered_order_items':
            out.setdefault('payment_method_id', out.get('payment_method_id', 'ORIGINAL_PAYMENT_METHOD'))

    # For modify_pending_order_items, ensure new_item_ids and payment_method_id
    if name == 'modify_pending_order_items':
        if 'new_item_ids' not in out or not out.get('new_item_ids'):
            item_ids = out.get('item_ids') or []
            out['new_item_ids'] = list(item_ids)
        out.setdefault('payment_method_id', 'ORIGINAL_PAYMENT_METHOD')

    # For modify_pending_order_payment, ensure payment_method_id exists (use placeholder if missing)
    if name == 'modify_pending_order_payment':
        out.setdefault('payment_method_id', out.get('payment_method_id', 'ORIGINAL_PAYMENT_METHOD'))

    # Ensure lists are lists
    for k, v in list(out.items()):
        if isinstance(v, tuple):
            out[k] = list(v)

    # Special mapping to synthesize 'summary' for transfer_to_human_agents
    if name == 'transfer_to_human_agents':
        if 'summary' not in out or not out.get('summary'):
            reason = out.get('reason', '')
            user_id = out.get('user_id', '')
            # Create a compact summary
            summary_parts = []
            if reason:
                summary_parts.append(f"Reason: {reason}")
            if user_id:
                summary_parts.append(f"User: {user_id}")
            out['summary'] = '; '.join(summary_parts) if summary_parts else 'Escalation requested'

    # Handle find_user_id_by_name_zip: support 'name' (full name) and extract zip if possible
    if name == 'find_user_id_by_name_zip':
        # Map common aliases
        if 'user_name' in out and 'name' not in out:
            out['name'] = out.pop('user_name')
        if 'location_zip' in out and 'zip' not in out:
            out['zip'] = out.pop('location_zip')

        # If 'name' provided, split into first/last
        full = out.get('name') or out.get('full_name') or out.get('customer_name')
        if full and ('first_name' not in out or 'last_name' not in out):
            parts = str(full).strip().split()
            if len(parts) >= 2:
                out.setdefault('first_name', parts[0])
                out.setdefault('last_name', parts[-1])
        # Extract 5-digit zip from provided strings if zip missing or non-numeric
        import re
        zip_val = out.get('zip')
        if not zip_val or not re.match(r"^\d{5}$", str(zip_val)):
            # try address fields and the 'query' if present
            for k in ('address1', 'address2', 'city', 'state', 'country', 'query'):
                if k in out and isinstance(out[k], str):
                    m = re.search(r"(\d{5})", out[k])
                    if m:
                        out['zip'] = m.group(1)
                        break

    # Normalize single item_id -> item_ids for item-based actions
    if 'item_id' in out and 'item_ids' not in out:
        out['item_ids'] = [out.pop('item_id')]



    # Optionally filter out keys that are not part of canonical expected params
    EXPECTED_PARAMS = {
        'find_user_id_by_email': {'email'},
        'find_user_id_by_name_zip': {'first_name', 'last_name', 'zip'},
        'get_order_details': {'order_id'},
        'get_user_details': {'user_id'},
        'modify_pending_order_address': {'order_id','address1','address2','city','state','country','zip'},
        'modify_user_address': {'user_id','address1','address2','city','state','country','zip'},
        'cancel_pending_order': {'order_id','reason'},
        'return_delivered_order_items': {'order_id','item_ids','payment_method_id'},
        'exchange_delivered_order_items': {'order_id','item_ids','new_item_ids','payment_method_id'},
        'modify_pending_order_items': {'order_id','item_ids','new_item_ids','payment_method_id'},
        'modify_pending_order_payment': {'order_id','payment_method_id'},
        'transfer_to_human_agents': {'summary'},
        'list_all_product_types': set(),
        'get_product_details': {'product_id'},
        'calculate': {'expression'},
        'think': set()
    }

    allowed = EXPECTED_PARAMS.get(name)
    if allowed is not None:
        # keep only allowed keys (and keep required ones already ensured)
        filtered_out = {k:v for k,v in out.items() if k in allowed}
        return filtered_out

    return out
