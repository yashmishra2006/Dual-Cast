# 🎙️ Dual Cast Commentary System

AI-powered real-time dual-caster commentary for Valorant matches — making every custom lobby feel like a professional esports broadcast.

## 🚀 Overview

The Dual Cast Commentary System brings the thrill of professional shoutcasting to casual and community Valorant games. Using AI, text-to-speech (TTS), and game event detection, the system generates **real-time dual voice commentary** — one analytical and one hype — for three phases of the game:
- 🧠 **Agent Selection Phase**
- 💰 **Buy Phase**
- 🔥 **Game Phase**

No casters? No problem.

## 🎯 Problem Statement

Amateur Valorant matches often lack engaging commentary, making spectating or reviewing gameplay less immersive. This project solves that by using AI to deliver live, energetic, and strategic commentary — fully automated.

## 🧠 Key Features

- 🎭 **Dual TTS Voices**: One for hype, one for analysis.
- ⏱️ **Real-Time Reactions**: Commentary triggers synced with game events.
- 🧩 **Modular Templates**: Customizable commentary lines for various scenarios.
- 🎮 **Valorant-Specific Logic**: Recognizes agent picks, spike events, aces, thriftys, clutches, and more.

## 🛠️ Tech Stack

- 🐍 Python
- 🗣️ TTS (e.g. Coqui TTS / Bark / ElevenLabs-compatible)
- 🎯 Game State Integration (via screenshot parsing or event feed)
- ⚙️ Flask/Streamlit for UI (optional)
- 🧠 Template system (CSV/JSON-based)

## 🧪 How It Works

1. **Agent Selection**: Detects locked agents and generates hype/analysis commentary.
2. **Buy Phase**: Comments on economy, buys, and loadouts.
3. **Game Phase**: Reacts to spike events, kills, clutches, etc.
4. **Dual Caster Engine**: Alternates between voices for dynamic commentary.

