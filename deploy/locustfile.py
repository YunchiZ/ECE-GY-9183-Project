from locust import HttpUser, task, between, events
import json, random, pathlib, os, logging, itertools

DATA_FILE = pathlib.Path("./tests/sample.jsonl")
SAMPLE_DOCS: list[str] = []
RESERVOIR_K = int(os.getenv("CORPUS_SAMPLE_SIZE", 10_000))

@events.init.add_listener
def _load_jsonl(env, **kw):
    global SAMPLE_DOCS
    if not DATA_FILE.exists():
        logging.error(f"[locust]  {DATA_FILE} does not exits, quit load test")
        env.runner.quit()
        return

    reservoir = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                doc = json.loads(line)["document"].strip()
            except Exception:
                continue
            if len(reservoir) < RESERVOIR_K:
                reservoir.append(doc)
            else:
                j = random.randrange(i)
                if j < RESERVOIR_K:
                    reservoir[j] = doc

    SAMPLE_DOCS = reservoir
    logging.info(f"[locust] obtains {len(SAMPLE_DOCS):,}/{i:,} samples "
                 f"(file {DATA_FILE.stat().st_size/1e6:.1f} MB)")

class ApiUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def predict(self):
        payload = {
            "text": random.choice(SAMPLE_DOCS),
            "prediction_id": random.randint(1, 9_999_999)
        }
        self.client.post("/predict", json=payload, name="/predict")