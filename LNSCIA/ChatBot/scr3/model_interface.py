# model_interface.py
import torch

class ModelInterface:
    def __init__(self, model_path: str, backend: str = "transformers", device: torch.device = None):
        """
        Inicializa a interface do modelo usando o backend especificado.
        backend: 'transformers' (padrão) ou 'llama_cpp' para modelos quantizados leves.
        """
        self.backend = backend
        self.device = device if device is not None else torch.device("cpu")
        if self.backend == "llama_cpp":
            try:
                from llama_cpp import Llama
                # Parâmetros básicos – podem ser ajustados conforme necessário.
                self.model = Llama(model_path=model_path, n_threads=4)
            except ImportError:
                raise ImportError("llama_cpp não está instalado. Instale-o ou escolha backend='transformers'.")
        elif self.backend == "transformers":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            # Garante que o pad_token está configurado corretamente.
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if self.device.type == "cpu":
                self.model.to(self.device)
        else:
            raise ValueError("Backend não suportado. Use 'llama_cpp' ou 'transformers'.")

    def generate_response(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Gera uma resposta baseada no prompt usando o backend escolhido.
        """
        if self.backend == "llama_cpp":
            # Exemplo de chamada com llama_cpp – os parâmetros podem variar conforme a implementação.
            response = self.model(prompt=prompt, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            return response["choices"][0]["text"].strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.0
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove o prompt da resposta gerada
            generated = response[len(prompt):].strip()
            if not generated or len(generated.split()) < 3:
                generated = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Como posso ajudar?"
            return generated
