import os
import json
from pathlib import Path
import asyncio
import aiofiles
import google.generativeai as genai
from google.generativeai.types import HarmCategory

class AsyncGEMINI:
    def __init__(self, api_key=None, model="gemini-2.5-flash-preview-04-17", concurrency=128, writer_workers=3):
        self.api_key = api_key or self.get_api_key()
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.contador = 0
        self.contador_lock = asyncio.Lock()

        genai.configure(api_key=self.api_key)

        self.queue = asyncio.Queue(maxsize=1000)  # Otimização: fila limitada
        self.writer_tasks = []
        self.writer_workers = writer_workers

        self.client = genai.GenerativeModel(
            model_name=self.model,
            safety_settings=[
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": 4},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": 4},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": 4},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": 4},
            ]
        )

    def get_api_key(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente GEMINI_API_KEY não foi definida.")
        return api_key.strip()

    async def async_init(self):
        Path("data/ids").mkdir(parents=True, exist_ok=True)
        self.writer_tasks = [
            asyncio.create_task(self.writer_loop())
            for _ in range(self.writer_workers)
        ]

    async def writer_loop(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break

            item_id = item["id"]
            try:
                json_path = Path(f"data/ids/{item_id}.json")
                if json_path.exists():
                    json_path.unlink()  # Remove antes para garantir truncamento

                async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                print(f"[{item_id}] ✅ Salvo (fila)")
            except Exception as e:
                print(f"[{item_id}] ❌ Erro ao salvar: {e}")

    async def shutdown(self):
        for _ in range(self.writer_workers):
            await self.queue.put(None)
        await asyncio.gather(*self.writer_tasks)

    async def invoke_and_save(self, item: dict):
        item_id = item["id"]
        prompt = item["text"]

        try:
            async with self.semaphore:
                response = await self.client.generate_content_async(prompt)

            if not response.candidates:
                reason = getattr(response.prompt_feedback, "block_reason", "DESCONHECIDO")
                print(f"[{item_id}] Prompt bloqueado (motivo: {reason})")
                return

            if not response.text:
                print(f"[{item_id}] Resposta vazia.")
                return

            text = response.text.strip()
            usage_data = {
                k: getattr(response.usage_metadata, k)
                for k in dir(response.usage_metadata)
                if not k.startswith("_") and not callable(getattr(response.usage_metadata, k))
            }

            await self.queue.put({
                "id": item_id,
                "text": text,
                "usage_metadata": usage_data,
                "perfil": item["perfil"],
                "source": item["source"],
                "model": self.model
            })

            async with self.contador_lock:
                self.contador += 1

        except Exception as e:
            print(f"[{item_id}] ❌ Erro ({type(e).__name__}): {e}")