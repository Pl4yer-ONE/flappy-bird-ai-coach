# Mistake Analyzer - Rule-Based Mistake Classification

"""
Analyzes player gameplay to classify mistakes using rule-based detection.
All thresholds are explicit and explainable for academic defense.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LATE_REACTION_THRESHOLD, OVERFLAP_WINDOW, PANIC_FLAP_COUNT,
    PANIC_WINDOW, POOR_ANTICIPATION_DISTANCE, POOR_CENTERING_THRESHOLD,
    STATE_HISTORY_LENGTH, PIPE_GAP
)


class MistakeType(Enum):
    """Classification of player mistakes."""
    PANIC_BEHAVIOR = "panic_behavior"
    OVERFLAPPING = "overflapping"
    LATE_REACTION = "late_reaction"
    POOR_ANTICIPATION = "poor_anticipation"
    POOR_CENTERING = "poor_centering"
    GENERAL = "general"  # Fallback category


@dataclass
class MistakeAnalysis:
    """Result of mistake analysis."""
    mistake_type: MistakeType
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    frame_index: int  # Frame where mistake was detected


class MistakeAnalyzer:
    """
    Analyzes player state history to classify the cause of death.
    
    Uses rule-based detection with explicit thresholds for:
    - Panic behavior (rapid flapping near obstacles)
    - Overflapping (too many flaps in short window)
    - Late reaction (flapping too close to pipe)
    - Poor anticipation (not positioning early)
    - Poor centering (not staying centered in gap)
    
    All thresholds are documented and explainable.
    """
    
    def __init__(self):
        """Initialize analyzer with configuration thresholds."""
        # Detection thresholds (from config)
        self.late_reaction_threshold = LATE_REACTION_THRESHOLD
        self.overflap_window = OVERFLAP_WINDOW
        self.panic_flap_count = PANIC_FLAP_COUNT
        self.panic_window = PANIC_WINDOW
        self.poor_anticipation_distance = POOR_ANTICIPATION_DISTANCE
        self.poor_centering_threshold = POOR_CENTERING_THRESHOLD
        self.history_length = STATE_HISTORY_LENGTH
        
    def analyze_death(self, state_history: List[Dict[str, Any]]) -> MistakeAnalysis:
        """
        Analyze state history to determine cause of death.
        
        Args:
            state_history: List of state dicts, last state is death
            
        Returns:
            MistakeAnalysis with type, confidence, and details
        """
        if not state_history:
            return MistakeAnalysis(
                mistake_type=MistakeType.GENERAL,
                confidence=0.0,
                details={'error': 'No state history provided'},
                frame_index=-1
            )
            
        # Get recent states for analysis
        recent_states = state_history[-self.history_length:]
        
        # Extract flap events
        flap_frames = self._detect_flaps(recent_states)
        
        # Run detection in priority order (most specific first)
        # 1. Panic behavior
        panic_result = self._detect_panic(recent_states, flap_frames)
        if panic_result:
            return panic_result
            
        # 2. Overflapping
        overflap_result = self._detect_overflapping(recent_states, flap_frames)
        if overflap_result:
            return overflap_result
            
        # 3. Late reaction
        late_result = self._detect_late_reaction(recent_states, flap_frames)
        if late_result:
            return late_result
            
        # 4. Poor anticipation
        anticipation_result = self._detect_poor_anticipation(recent_states)
        if anticipation_result:
            return anticipation_result
            
        # 5. Poor centering
        centering_result = self._detect_poor_centering(recent_states)
        if centering_result:
            return centering_result
            
        # Fallback
        return MistakeAnalysis(
            mistake_type=MistakeType.GENERAL,
            confidence=0.5,
            details={'message': 'Could not determine specific cause'},
            frame_index=len(state_history) - 1
        )
        
    def _detect_flaps(self, states: List[Dict]) -> List[int]:
        """
        Detect frames where flaps occurred.
        
        A flap is detected when time_since_last_flap resets to 0.
        """
        flap_frames = []
        for i, state in enumerate(states):
            if state.get('time_since_last_flap', 1) == 0:
                flap_frames.append(i)
        return flap_frames
        
    def _detect_panic(self, states: List[Dict], flap_frames: List[int]) -> Optional[MistakeAnalysis]:
        """
        Detect panic behavior: 3+ flaps in 15 frames near obstacles.
        
        Panic is characterized by rapid, erratic flapping when stressed.
        """
        if len(states) < self.panic_window:
            return None
            
        # Check last panic_window frames
        recent_flaps = [f for f in flap_frames if f >= len(states) - self.panic_window]
        
        if len(recent_flaps) >= self.panic_flap_count:
            # Check if near pipe (adds to panic classification)
            final_distance = states[-1].get('distance_to_next_pipe', float('inf'))
            near_pipe = final_distance < 100
            
            confidence = min(0.5 + 0.1 * (len(recent_flaps) - self.panic_flap_count), 0.95)
            if near_pipe:
                confidence = min(confidence + 0.2, 0.95)
                
            return MistakeAnalysis(
                mistake_type=MistakeType.PANIC_BEHAVIOR,
                confidence=confidence,
                details={
                    'flap_count': len(recent_flaps),
                    'window_size': self.panic_window,
                    'near_pipe': near_pipe,
                    'distance_to_pipe': final_distance
                },
                frame_index=recent_flaps[0] if recent_flaps else len(states) - 1
            )
            
        return None
        
    def _detect_overflapping(self, states: List[Dict], flap_frames: List[int]) -> Optional[MistakeAnalysis]:
        """
        Detect overflapping: multiple flaps within 10-frame window.
        
        Overflapping leads to hitting the top pipe.
        """
        if len(flap_frames) < 2:
            return None
            
        # Find consecutive flaps that are too close together
        overflap_sequences = []
        for i in range(1, len(flap_frames)):
            interval = flap_frames[i] - flap_frames[i - 1]
            if interval <= self.overflap_window:
                overflap_sequences.append((flap_frames[i - 1], flap_frames[i], interval))
                
        if overflap_sequences:
            # Calculate severity
            min_interval = min(s[2] for s in overflap_sequences)
            confidence = 0.6 + 0.1 * (self.overflap_window - min_interval)
            confidence = max(0.5, min(confidence, 0.9))
            
            # Check if bird was going up (confirming overflap)
            final_velocity = states[-1].get('bird_velocity', 0)
            going_up = final_velocity < 0
            
            return MistakeAnalysis(
                mistake_type=MistakeType.OVERFLAPPING,
                confidence=confidence,
                details={
                    'sequences': overflap_sequences,
                    'min_interval': min_interval,
                    'threshold': self.overflap_window,
                    'going_up': going_up,
                    'final_velocity': final_velocity
                },
                frame_index=overflap_sequences[-1][1]
            )
            
        return None
        
    def _detect_late_reaction(self, states: List[Dict], flap_frames: List[int]) -> Optional[MistakeAnalysis]:
        """
        Detect late reaction: flapping when < 50 pixels from pipe.
        
        By this point, there's not enough time to adjust trajectory.
        """
        for flap_frame in reversed(flap_frames):
            if flap_frame < len(states):
                distance = states[flap_frame].get('distance_to_next_pipe', float('inf'))
                if distance < self.late_reaction_threshold:
                    confidence = 0.7 + 0.2 * (1 - distance / self.late_reaction_threshold)
                    return MistakeAnalysis(
                        mistake_type=MistakeType.LATE_REACTION,
                        confidence=confidence,
                        details={
                            'flap_distance': distance,
                            'threshold': self.late_reaction_threshold,
                            'flap_frame': flap_frame
                        },
                        frame_index=flap_frame
                    )
                    
        return None
        
    def _detect_poor_anticipation(self, states: List[Dict]) -> Optional[MistakeAnalysis]:
        """
        Detect poor anticipation: not centering when far from pipe.
        
        Should start positioning when 150+ pixels away.
        """
        # Check early states (when far from pipe)
        early_states = states[:len(states) // 2]
        
        for state in early_states:
            distance = state.get('distance_to_next_pipe', 0)
            if distance > self.poor_anticipation_distance:
                bird_y = state.get('bird_y', 0)
                gap_y = state.get('next_pipe_gap_y', 0)
                vertical_distance = abs(bird_y - gap_y)
                
                # Should be within half the gap of center
                if vertical_distance > PIPE_GAP / 2.5:
                    return MistakeAnalysis(
                        mistake_type=MistakeType.POOR_ANTICIPATION,
                        confidence=0.6,
                        details={
                            'vertical_distance': vertical_distance,
                            'distance_to_pipe': distance,
                            'bird_y': bird_y,
                            'gap_y': gap_y
                        },
                        frame_index=states.index(state)
                    )
                    
        return None
        
    def _detect_poor_centering(self, states: List[Dict]) -> Optional[MistakeAnalysis]:
        """
        Detect poor centering: average distance from gap center is too high.
        """
        distances = []
        for state in states:
            bird_y = state.get('bird_y', 0)
            gap_y = state.get('next_pipe_gap_y', bird_y)  # Default to bird_y if no pipe
            if gap_y > 0:  # Valid pipe data
                distances.append(abs(bird_y - gap_y))
                
        if distances:
            avg_distance = sum(distances) / len(distances)
            if avg_distance > self.poor_centering_threshold:
                return MistakeAnalysis(
                    mistake_type=MistakeType.POOR_CENTERING,
                    confidence=0.55,
                    details={
                        'average_distance': avg_distance,
                        'threshold': self.poor_centering_threshold,
                        'sample_count': len(distances)
                    },
                    frame_index=len(states) - 1
                )
                
        return None
        
    def get_threshold_documentation(self) -> Dict[str, str]:
        """Get documentation for all thresholds (for viva defense)."""
        return {
            'late_reaction_threshold': f"{self.late_reaction_threshold}px - At 3px/frame velocity, gives ~16 frames to react",
            'overflap_window': f"{self.overflap_window} frames - Minimum time for bird to fall meaningfully",
            'panic_flap_count': f"{self.panic_flap_count} flaps - Statistical outlier in normal play",
            'panic_window': f"{self.panic_window} frames - Window for panic detection",
            'poor_anticipation_distance': f"{self.poor_anticipation_distance}px - ~50 frames ahead, reasonable planning horizon"
        }
