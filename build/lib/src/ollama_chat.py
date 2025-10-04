from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class DeepSeekChat:
    def __init__(self,
                 model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 device="mps"):
        """
        Load DeepSeek model on Mac (MPS). 
        No bitsandbytes / CUDA required.
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading model... (may take a while first time)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float16
        )

        print("Building pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

    def chat(self, prompt: str) -> str:
        out = self.pipe(prompt)[0]["generated_text"]
        return out[len(prompt):].strip()

deepseek = DeepSeekChat()

def chat_with_deepseek(prompt: str) -> str:
    return deepseek.chat(prompt)