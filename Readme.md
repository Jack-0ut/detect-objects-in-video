# AI Video Processor CLI

A professional, easy-to-use command-line toolkit for AI-powered video object detection, metadata search, and frame extraction.  
Supports local video files and YouTube URLs.

---

## Features

- **Process videos** with YOLOv8 object detection and tracking.
- **Automatic metadata extraction** and storage for every run.
- **Semantic search**: Find frames by natural language queries (e.g., "person walking").
- **Frame extraction**: Instantly display or save annotated frames matching your search.
- **Supports**: Local files, YouTube links.

---

## Installation

1. **Clone the repository** (or install from PyPI if published):

    ```bash
    git clone https://github.com/Jack-0ut/ai-detection.git
    cd ai-detection
    ```

2. **Install dependencies** (preferably in a virtual environment):

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### **Basic Processing**

Process a video (local file or YouTube URL):

```bash
python cli.py "video.mp4"
python cli.py "https://www.youtube.com/watch?v=GBkJY86tZRE"
```

### **Process and Search**

Process and immediately search for frames matching a query:

```bash
python cli.py "video.mp4" --query "person walking"
```

### **Show or Save Result Frame**

Display the first matching frame in a window:

```bash
python cli.py "video.mp4" --query "person walking" --show-frame
```

Save the first matching frame as an image:

```bash
python cli.py "video.mp4" --query "person walking" --save-frame result.jpg
```

### **All Options**

```bash
python cli.py --help
```

---

## Output Organization

- All outputs (logs, metadata, etc.) are stored in a timestamped folder under `outputs/` for each run.
- Metadata and logs are automatically managed for you.

---

## Graphical User Interface

You can also use the interactive UI by running:

```bash
streamlit run ui.py
```

This will launch a web-based interface for uploading videos, processing, searching, and viewing results visually.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

---

## FAQ

**Q: Can I use this with YouTube videos?**  
A: Yes! Just pass the YouTube URL as the source.

**Q: Can I use my webcam?**  
A: No, webcam input is not supported in this version.

**Q: Where are my results stored?**  
A: In a unique folder under `outputs/` for each run.

---

## License

MIT License

---

## Contributing

Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

---

## Authors

- [Your Name](https://github.com/Jack-0ut)

---

**Enjoy AI-powered video analysis from your terminal or browser!**