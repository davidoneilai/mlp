import asyncio, random, json, time, argparse
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from PIL import Image
from gemini import AsyncGEMINI

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
N_SAMPLES     = 1000
BATCH_SIZE    = 500
CONCURRENCY   = 10000
MODEL_NAME    = "gemini-2.5-flash-preview-04-17"
DATASET_SPLIT = "test"
JSONL_PATH    = Path("data/label_corrections.jsonl")
IDS_DIR       = Path("data/ids_labels")

def build_prompt(label: str) -> str:
    classes = ", ".join(CLASS_NAMES)
    return (
        f"A imagem abaixo está rotulada como **{label}**.\n"
        f"Isso está correto?\n"
        f"- Se SIM, responda exatamente: CORRECT\n"
        f"- Se NÃO, responda apenas com o rótulo correto entre: {classes}\n"
    )

def carregar_ids_jsonl(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {str(json.loads(l)["id"]) for l in f if l.strip() and '"id"' in l}

async def process_sample(idx, ds, ids_processados, semaphore, model):
    id_str = f"img_{idx}"
    if id_str in ids_processados:
        return
    try:
        image = ds[idx]["image"].convert("RGB")
        label_idx = ds[idx]["label"]
        prompt = build_prompt(CLASS_NAMES[label_idx])

        async with semaphore:
            await model.invoke_and_save({
                "id": id_str,
                "text": prompt,
                "image": image,
                "perfil": "label_correction",
                "source": "fashion_mnist"
            })
        ids_processados.add(id_str)
        image.close()
    except Exception as e:
        print(f"[{id_str}] Erro: {e}", flush=True)

async def process_block(indices, block_num, ds, ids_processados, semaphore, gemini):
    print(f"[Bloco {block_num}] {len(indices)} amostras...", flush=True)
    start = time.time()
    await asyncio.gather(*[
        process_sample(idx, ds, ids_processados, semaphore, gemini)
        for idx in indices
    ])
    print(f"Bloco {block_num} concluído ({time.time() - start:.2f}s)", flush=True)

async def main(total_parts=1, part=0):
    load_dotenv()
    gemini = AsyncGEMINI(model=MODEL_NAME, concurrency=CONCURRENCY)
    await gemini.async_init()

    ds = load_dataset("fashion_mnist", split=DATASET_SPLIT)
    rng = random.Random(42)
    all_indices = rng.sample(range(len(ds)), N_SAMPLES)

    indices_this_part = [idx for i, idx in enumerate(all_indices) if i % total_parts == part]
    ids_processados = carregar_ids_jsonl(JSONL_PATH)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    bloco_num, buffer = 1, []
    try:
        for idx in indices_this_part:
            if str(idx) in ids_processados:
                continue
            buffer.append(idx)
            if len(buffer) >= BATCH_SIZE:
                await process_block(buffer, bloco_num, ds, ids_processados, semaphore, gemini)
                buffer.clear()
                bloco_num += 1

        if buffer:
            await process_block(buffer, bloco_num, ds, ids_processados, semaphore, gemini)

        print("Fim da verificação.")
    finally:
        await gemini.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_parts", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()

    asyncio.run(main(total_parts=args.total_parts, part=args.part))
