from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import time
import json
import traceback

app = FastAPI()

# ========= model loading =========
MODEL_PATH = "outputs/tau_bench_rl/merged_model"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# pad_token fallback (very important)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# ========= OpenAI API Schema =========
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: int = 1024
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# ========= ChatCompletion =========
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 🔥 Key: remove token_type_ids
        inputs.pop("token_type_ids", None)

        # generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # decode
        gen_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # ===== tool calling parsing (for tau-bench) =====
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": gen_text
        }
        finish_reason = "stop"

        # Allow model to directly output JSON tool call
        if gen_text.startswith("{") and "tool_calls" in gen_text:
            try:
                tool_payload = json.loads(gen_text)
                message["content"] = None
                message["tool_calls"] = tool_payload["tool_calls"]
                finish_reason = "tool_calls"
            except Exception:
                pass

        return ChatResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }]
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ========= Model List =========
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "gpt-oss-20b-tau",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user"
        }]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
