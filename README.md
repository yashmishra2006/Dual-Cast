# 🎙️ Dual Cast Commentary System

AI-powered real-time dual-caster commentary for Valorant matches — built for Code for Bharat Season 2.

## 🚀 Overview

The Dual Cast Commentary System brings the thrill of professional shoutcasting to casual and community Valorant games. Using AI, text-to-speech (TTS), and game event detection, the system generates **real-time dual voice commentary** — one analytical and one hype — for three phases of the game:
- 🧠 **Agent Selection Phase**
- 💰 **Buy Phase**
- 🔥 **Game Phase**

No casters? No problem.

## 🧠 Architecture & Workflow

1. **🎞️ Frame Processor (Main Script)**  
   - Processes incoming video stream.
   - Extracts 1 frame per second and sends it to the **Phase Classifier**.

2. **📊 Phase Classifier**  
   - Classifies the current frame into one of the three phases:
     - **Agent Selection**
     - **Buy Phase**
     - **Game Phase**

3. **🧠 Phase-Specific Model Pipelines**
   - Based on the phase, frames are routed to dedicated model stacks:
     - 🔹 **Agent Phase** → `ResNet-50`
     - 🔹 **Buy Phase** → `YOLOv8 + ResNet-18`
     - 🔹 **Game Phase** → `YOLOv8 + ResNet-18`

4. **🔄 Back to Main Script**
   - Outputs (agents, weapons, spike status, etc.) are returned to the main script.
   - Triggers commentary templates and sends to TTS engine.

5. **🗣️ Dual-Caster Commentary Engine**
   - Two unique TTS voices simulate professional casters.
   - One delivers **hype**, the other provides **analysis**.
   - Voices alternate naturally with contextual awareness.

## 💡 Key Features

- 🎭 **Dual TTS Voices**: One hype, one analytical — like real shoutcasters.
- ⏱️ **Real-Time Reactions**: Commentary matches the game flow with low latency.
- 🧩 **Modular Templates**: Custom JSON/CSV commentary lines for all match events.
- 🤖 **Smart Phase Detection**: Uses a custom classifier to detect game phases.
- 🧠 **Agent & Event Recognition**: Leverages ResNet and YOLOv8 to detect agent locks, spike, clutches, aces, buys, and more.
- 🎮 **Valorant-Specific Logic**: Built entirely around Valorant game mechanics.
- 📦 **Plug-and-Play UI (optional)**: Simple Flask or Streamlit front-end for demo/testing.

## 📦 Folder Structure
dual-cast/
├── main.py # Main orchestration script
├── phase_classifier/ # Predicts current game phase
├── agent_phase/ # ResNet-50-based agent detection
├── buy_phase/ # YOLOv8 + ResNet-18 for buy detection
├── game_phase/ # YOLOv8 + ResNet-18 for gameplay events
├── tts_engine/ # Dual voice TTS generator
├── commentary_templates/ # Phase-wise JSON/CSV commentary lines
└── utils/ # Shared helpers (frame extraction, etc.)


## 🧠 Models Used

| Phase           | Models Used            |
|----------------|------------------------|
| Agent Phase     | ResNet-50              |
| Buy Phase       | YOLOv8 + ResNet-18     |
| Game Phase      | YOLOv8 + ResNet-18     |
| Phase Detection | Custom Phase Classifier|

## 🎯 Use Cases

- 🎮 Valorant custom matches & scrims
- 📺 Twitch/YouTube live overlays
- 🧠 Post-match review with voice highlights
- ✂️ Auto-highlight reels with caster voices

## 🛠️ Tech Stack

- 🐍 Python
- 🎯 PyTorch (YOLOv8, ResNet-18/50)
- 🗣️ TTS Engine (Coqui TTS / Bark / ElevenLabs-compatible)
- 🖼️ OpenCV
- 🧪 Optional: Streamlit/Flask UI for testing

## 🏁 Getting Started

```bash
git clone https://github.com/your-team/dual-cast.git
cd dual-cast
pip install -r requirements.txt
python main.py --video gameplay.mp4

