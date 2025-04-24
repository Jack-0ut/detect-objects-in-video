import streamlit as st
from processing_engine import ProcessingEngine
import tempfile

st.set_page_config(page_title="AI Video Processor", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: white;'>🎥 AI Video Processor</h1>",
    unsafe_allow_html=True,
)

st.markdown("##### Choose a video source:")
video_source = st.radio("Video Source", ["YouTube URL", "Upload from Device"], horizontal=True)

video_path = None
if video_source == "YouTube URL":
    youtube_url = st.text_input("Paste YouTube URL here:")
    if youtube_url:
        video_path = youtube_url

else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file.close()
        video_path = temp_file.name

# Trigger processing
if st.button("🚀 Process Video"):
    if not video_path:
        st.error("Please provide a valid video input.")
    else:
        st.write("🔍 Processing video, please wait...")
        st.session_state.engine = ProcessingEngine(video_path)
        st.session_state.engine.run()
        st.success("✅ Video processed successfully!")

# Search interface
st.markdown("---")
st.markdown("#### Search Something in the Processed Video:")
query = st.text_input("Enter a query (e.g., 'person walking'):")

if st.button("🔎 Search Metadata"):
    if not query:
        st.error("Please enter a valid query.")
    elif "engine" not in st.session_state:
        st.error("Process a video first before searching.")
    else:
        engine = st.session_state.engine
        st.write(f"Searching for: '{query}'...")
        results = engine.search_metadata(query)

        if results:
            st.success(f"🔍 Found {len(results)} results.")
            frame_num = results[0]['frames'][0]
            frame = engine.extract_frame_with_annotations(frame_num)
            st.image(frame, caption=f"Frame {frame_num} with annotations", use_container_width=True)
        else:
            st.warning(f"No results found for '{query}'.")

