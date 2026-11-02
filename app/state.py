"""
Shared runtime state for the FastAPI app — the OpenAI client and the two
separate long-lived ChromaDB clients (one for VIQ, one for RISQ).

IMPORTANT — how to use this module correctly:
Always import the module itself and read/write attributes on it:

    from app import state
    state.viq_chroma_client = chromadb.PersistentClient(...)
    collection = state.collection

Never do `from app.state import viq_chroma_client` and then reassign
`viq_chroma_client = ...` in your own module — that only rebinds your
local name and the rest of the app would keep seeing the old value. This
is exactly the kind of bug that caused RISQ training writes to be
invisible to search until a restart (a stale/duplicate client
reference); this module exists so there is exactly ONE place holding
these handles.

VIQ and RISQ each get their OWN ChromaDB PersistentClient pointed at a
separate on-disk directory (data/viq_vectordb vs data/risq_vectordb) —
not just separate collections inside a shared client. This is a
deliberate physical isolation, not just logical: a bug in one module's
Chroma usage cannot corrupt or even see the other module's directory.
"""

# OpenAI client (VIQ and RISQ both use this same client/model — there is
# only one OpenAI account/key involved, no isolation concern here)
client = None

# ===== VIQ — separate ChromaDB store, separate collection handles =====
viq_chroma_client = None
collection = None  # the viq_questions collection

# ===== RISQ — separate ChromaDB store, separate collection handles =====
risq_chroma_client = None
risq_collection_questions = None
risq_collection_training = None
risq_collection_documents = None
