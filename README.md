# Crop Disease Detection Web Portal
This project is a FastAPI-based web application that allows users to upload crop videos, extract frames, and analyze them using a pre-trained Vision Transformer (ViT) model to predict potential leaf diseases (supports rice, wheat, and corn).

**🚀 Features**
- Upload crop videos from a web interface.
- Automatically extract frames at regular intervals.
- Run AI-based disease predictions on extracted frames.
- View results in the browser.
- Option to clear uploaded videos and analysis frames.

**✅ Requirements**
- Python: 3.10–3.11 recommended
- pip: Latest version
- Internet connection (to download the pre-trained model on first run)

**📥 Installation**
1. Clone the repository
- git clone https://github.com/rakibHridoy206/crop_disease_portal.git
- cd crop_disease_portal

2. Create a virtual environment
* On Windows (PowerShell)
  - python -m venv venv
  - venv\Scripts\activate
* On macOS/Linux (zsh/bash)
  - python3 -m venv venv
  - source venv/bin/activate

3. Install dependencies
- pip install --upgrade pip
- pip install -r requirements.txt

**▶️ Run the Application**
- uvicorn main:app --reload

- Then open your browser and go to: http://127.0.0.1:8000/

**🧹 Clearing Data**
- In the web interface, use the Clear Data button to remove all uploaded videos and frames.

**⚠️ Notes**
- On the first run, model weights will be automatically downloaded (may take time).
- Ensure you have OpenCV and Pillow installed (included in requirements.txt).
- For Windows users behind slow networks, pip might timeout—re-run the installation if needed.
