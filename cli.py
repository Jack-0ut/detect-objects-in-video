import argparse
from processing_engine import ProcessingEngine

def process_video(source, query=None, topk=5, show_frame=False, save_frame=None):
    engine = ProcessingEngine(source)
    engine.run()
    print("Video processed and metadata saved.")

    if query:
        results = engine.search_metadata(query, k=topk)
        if results:
            print(f"\nTop {len(results)} results for query: '{query}'\n")
            for idx, res in enumerate(results):
                print(f"{idx+1}. Description: {res['description']}, Frames: {res['frames']}")
            # Show or save the first frame with annotations if requested
            frame_num = results[0]['frames'][0]
            frame = engine.extract_frame_with_annotations(frame_num)
            if show_frame:
                import cv2
                cv2.imshow(f"Frame {frame_num} with annotations", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save_frame:
                import cv2
                cv2.imwrite(save_frame, frame)
                print(f"Frame {frame_num} saved to {save_frame}")
        else:
            print(f"No results found for query: '{query}'")

def main():
    parser = argparse.ArgumentParser(description="AI Video Processor CLI")
    parser.add_argument("source", help="Video source (file path or YouTube URL)")
    parser.add_argument("--query", help="Search query after processing")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--show-frame", action="store_true", help="Show the first result frame with annotations")
    parser.add_argument("--save-frame", help="Path to save the first result frame image")

    args = parser.parse_args()
    process_video(
        args.source,
        query=args.query,
        topk=args.topk,
        show_frame=args.show_frame,
        save_frame=args.save_frame
    )

if __name__ == "__main__":
    main()