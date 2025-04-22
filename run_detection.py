from processing_engine import ProcessingEngine

engine = ProcessingEngine("https://www.youtube.com/watch?v=GBkJY86tZRE")
engine.run()
query = "person walking"
results = engine.search_metadata(query)

if results:
    print(f"\nTop {len(results)} results for query: '{query}'\n")
    frame_num = results[0]['frames'][0]
    frame = engine.extract_frame_with_annotations(frame_num)
    engine.show_frame(frame)

else:
    print(f"No results found for query: '{query}'")