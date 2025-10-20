import os, glob, re, uuid
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "knowledge_base"


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_SIZE = 384

def read_texts(folder="knowledge"):
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in [".txt", ".md"]:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((os.path.basename(p), f.read()))
    return docs

def chunk_text(text, max_len=900, overlap=150):
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    chunks, buf = [], []
    cur_len = 0
    for s in sents:
        sl = len(s)
        if cur_len + sl > max_len and buf:
            chunks.append(" ".join(buf).strip())
            back = " ".join(buf)[-overlap:]
            buf, cur_len = ([back], len(back))
        buf.append(s)
        cur_len += sl + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if len(c) > 50]

def main():
    docs = read_texts("knowledge_public")
    if not docs:
        print("⚠️ Папка 'knowledge_public/' порожня або файли не .txt/.md")
        return

    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    points = []
    total_chunks = 0
    for fname, content in docs:
        chunks = chunk_text(content)
        if not chunks:
            continue
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        for txt, vec in zip(chunks, embeddings):
            pid = str(uuid.uuid4())
            points.append(PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={"source": fname, "text": txt}
            ))
        total_chunks += len(chunks)

    if not points:
        print("⚠️ Нема що індексувати після нарізки.")
        return

    BATCH = 200
    for i in tqdm(range(0, len(points), BATCH), desc="Upserting to Qdrant"):
        client.upsert(collection_name=COLLECTION, points=points[i:i+BATCH])

    print(f"✅ Завантажено {total_chunks} фрагментів у колекцію '{COLLECTION}'.")

if __name__ == "__main__":
    main()
