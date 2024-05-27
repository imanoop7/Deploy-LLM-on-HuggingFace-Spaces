from ctransformers import AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel


llm = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF",
model_type='mistral',
max_new_tokens = 1096,
threads = 3,
)

#Pydantic object
class validation(BaseModel):
    prompt: str
#Fast API
app = FastAPI()

@app.post("/llm_on_cpu")
async def stream(item: validation):
    system_prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
    E_INST = "</s>"
    user, assistant = "<|user|>", "<|assistant|>"
    prompt = f"{system_prompt}{E_INST}\n{user}\n{item.prompt}{E_INST}\n{assistant}\n"
    return llm(prompt)