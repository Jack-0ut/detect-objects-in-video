from store_metadata import MetadataStore

store = MetadataStore(save_path="tracking_metadata.pkl")
query = "Person walking"

results = store.search(query, k=5)

print(f"\nTop {len(results)} results for query: '{query}'\n")
for i, res in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"  Description   : {res['description']}")
    print(f"  First Frame   : {res['frames'][0]}")
    print(f"  Total Frames  : {len(res['frames'])}")
    print(f"  Timestamp     : {res['timestamp']}")
    print(f"  Inference Time: {res['inference_time_ms']} ms")
    print("-" * 40)
