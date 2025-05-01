import os
import time
import threading
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel

MODEL_CLASSES = {
    "gcn": GCNModel,
    "gat": GATModel,
    "lightgcn": LightGCNModel,
    "graphsage": GraphSAGEModel
}

last_mod_times = {}


def watch_model_files(models, model_dir, interval=10):
    def reload_loop():
        while True:
            for name, model in models.items():
                path = os.path.join(model_dir, f"{name}_spotify.pt")
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if last_mod_times.get(name) != mtime:
                        print(f"[AutoReload] Reloading {name} model from disk")
                        models[name] = MODEL_CLASSES[name].load(path)
                        last_mod_times[name] = mtime
            time.sleep(interval)

    thread = threading.Thread(target=reload_loop, daemon=True)
    thread.start()