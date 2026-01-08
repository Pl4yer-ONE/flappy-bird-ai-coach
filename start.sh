#!/bin/bash

# ============================================================================
# Flappy Bird AI Coach - Unified Launch Script
# ============================================================================
# This script starts everything needed for the full AI coaching experience:
# - Ollama (LLM backend with Llama + LLaVA)
# - Flappy Bird AI Coach application
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     🎮 Flappy Bird AI Coach - Full Launch                 ║"
echo "║     Llama + LLaVA + DQN RL + gTTS Voice                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ===============================
# 1. Check and Start Ollama
# ===============================
echo -e "${YELLOW}📡 Checking Ollama service...${NC}"

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama not found. Please install from https://ollama.ai${NC}"
    echo "   Run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null && ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}🔄 Starting Ollama service...${NC}"
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    echo -e "${GREEN}✓ Ollama started (PID: $OLLAMA_PID)${NC}"
else
    echo -e "${GREEN}✓ Ollama already running${NC}"
fi

# ===============================
# 2. Check Required Models
# ===============================
echo -e "${YELLOW}🤖 Checking AI models...${NC}"

check_model() {
    local model=$1
    local display_name=$2
    
    if ollama list 2>/dev/null | grep -q "$model"; then
        echo -e "${GREEN}  ✓ $display_name ready${NC}"
        return 0
    else
        echo -e "${YELLOW}  ⏳ Pulling $display_name (this may take a few minutes)...${NC}"
        if ollama pull "$model" 2>/dev/null; then
            echo -e "${GREEN}  ✓ $display_name installed${NC}"
            return 0
        else
            echo -e "${RED}  ⚠ Could not pull $display_name${NC}"
            return 1
        fi
    fi
}

# Check for Llama (chat)
check_model "llama3:latest" "Llama 3 (Chat)" || check_model "llama3.2:latest" "Llama 3.2" || echo -e "${YELLOW}  Using fallback responses for chat${NC}"

# Check for LLaVA (vision)
check_model "llava:latest" "LLaVA (Vision)" || check_model "llava:7b" "LLaVA 7B" || echo -e "${YELLOW}  Vision coaching disabled${NC}"

# ===============================
# 3. Setup Python Environment
# ===============================
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"

# Try to activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment found, using system Python${NC}"
fi

# Quick dependency check
python -c "import pygame, torch, flappy_bird_gymnasium" 2>/dev/null || {
    echo -e "${YELLOW}📦 Installing missing dependencies...${NC}"
    pip install -q -r requirements.txt 2>/dev/null || echo "Some deps may be missing"
}

# ===============================
# 4. Launch the Application
# ===============================
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  🚀 Launching Flappy Bird AI Coach...                     ║${NC}"
echo -e "${CYAN}║                                                           ║${NC}"
echo -e "${CYAN}║  Features Enabled:                                        ║${NC}"
echo -e "${CYAN}║  • 🤖 Llama 3 - AI Chat Coach                            ║${NC}"
echo -e "${CYAN}║  • 👁️  LLaVA - Vision-based coaching                      ║${NC}"
echo -e "${CYAN}║  • 🎯 DQN RL Agent - Watch AI play                       ║${NC}"
echo -e "${CYAN}║  • 🔊 gTTS - Voice feedback                              ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Launch the main application
python main.py

echo -e "${GREEN}👋 Thanks for playing Flappy Bird AI Coach!${NC}"
