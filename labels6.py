import asyncio, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from gemini import AsyncGEMINI      # seu wrapper
from PIL import Image

# ───────────── Configurações ───────────── #
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
LABEL_OF_INTEREST = 6              # Shirt
CONCURRENCY       = 512            # ajuste p/ sua quota
MODEL_NAME        = "gemini-2.5-flash-preview-04-17"
DATASET_SPLIT     = "test"
OUT_JSONL         = Path("data/fmnist_shirt_verif.jsonl")
IDS_DIR           = Path("data/ids_labels")   # onde o wrapper cria JSONs

def build_prompt(label="Shirt") -> str:
    classes = ", ".join(CLASS_NAMES)
    return (
        f"A imagem abaixo está rotulada como **{label}**.\n"
        f"Isso está correto?\n"
        f"- Se SIM, responda exatamente: CORRECT\n"
        f"- Se NÃO, responda apenas com o rótulo correto entre: {classes}\n"
    )

async def process(idx, sample, sem, model, done):
    img_id = f"img_{idx}"
    if img_id in done:
        return
    prompt = build_prompt()
    try:
        async with sem:
            await model.invoke_and_save(
                {
                    "id": img_id,
                    "text": prompt,
                    "image": sample["image"].convert("RGB"),
                    "perfil": "fmnist_shirt_verification",
                    "source": "fashion_mnist"
                }        # << grava direto no JSONL agregado
            )
        done.add(img_id)
    except Exception as e:
        print(f"[{img_id}] erro: {e}", flush=True)

async def main():
    load_dotenv()
    gemini = AsyncGEMINI(
        model=MODEL_NAME,
        concurrency=CONCURRENCY,
        api_key=""
    )
    await gemini.async_init()

    ds = load_dataset("fashion_mnist", split=DATASET_SPLIT)
    shirt_idxs = [i for i, ex in enumerate(ds) if ex["label"] == LABEL_OF_INTEREST]
    print(f"{len(shirt_idxs)} imagens 'Shirt' para verificar.")

    #  ids já processados (caso exista um JSONL parcial)
    done = set()
    if OUT_JSONL.exists():
        with OUT_JSONL.open() as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    sem   = asyncio.Semaphore(CONCURRENCY)
    start = time.time()
    await asyncio.gather(*[
        process(i, ds[i], sem, gemini, done) for i in shirt_idxs
    ])
    print(f"Concluído em {time.time() - start:.1f}s")

    await gemini.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
