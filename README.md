# Flappy Bird AI Coach 🎮🤖

> **Complete AI Coaching System** integrating **Llama**, **LLaVA**, **DQN**, and **gTTS**

A BTech final-year project that combines **Reinforcement Learning** and **Explainable AI** to create an intelligent coaching system for Flappy Bird.

## 🌟 Features

- **🎮 OG Flappy Bird** - Original physics and gameplay
- **🧠 Dueling Double DQN** - Superhuman AI agent
- **🎓 AI Coaching** - Rule-based mistake analysis
- **💬 Llama LLM** - Natural language feedback
- **👁️ LLaVA Vision** - Screenshot analysis
- **🔊 gTTS Voice** - Spoken coaching advice

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with menu
python main.py

# Or specify mode directly
python main.py --mode play      # Play without coach
python main.py --mode coach     # Play with AI coach
python main.py --mode nlp       # Play with LLM coach
python main.py --mode ai        # Watch AI play
python main.py --mode train     # Train DQN agent
python main.py --mode arcade    # Retro arcade edition
```

## 🎯 Game Modes

| Mode | Description |
|------|-------------|
| **Human Play** | Classic Flappy Bird gameplay |
| **With AI Coach** | Get feedback on mistakes |
| **With NLP Coach** | Llama-enhanced natural language coaching |
| **Watch AI** | See trained DQN agent play |
| **Training** | Train your own agent |
| **Arcade** | Retro arcade edition |

## 📁 Project Structure

```
flappy_bird_ai_coach/
├── main.py              # Entry point
├── config.py            # Configuration
├── game/                # Game engine
├── rl_agent/            # DQN agent
├── coach/               # AI coaching
├── llm/                 # Llama integration
├── vision/              # LLaVA integration
├── tts/                 # Voice coaching
├── ui/                  # Menu & HUD
└── models/              # Trained models
```

## 🧠 AI Coach Mistake Types

1. **Panic Behavior** - Rapid flapping near obstacles
2. **Overflapping** - Too many flaps in quick succession
3. **Late Reaction** - Flapping too close to pipes
4. **Poor Anticipation** - Not positioning early enough
5. **Poor Centering** - Not staying centered in gaps

## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Pygame 2.5+
- Ollama (for LLM/Vision features)

## 🎓 Academic Defense

All thresholds and decisions are documented and explainable. See `coach/mistake_analyzer.py` for threshold justifications.

## 📝 License

MIT License - Built for educational purposes.
