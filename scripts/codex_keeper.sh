#!/bin/bash
# Codex Keeper - monitors soxl-codex tmux session and urges it to keep working

SESSION="soxl-codex"
WINDOW="0"
PANE="0"

while true; do
    sleep 300  # check every 5 minutes
    
    # Capture last few lines of the pane
    LAST_LINES=$(tmux capture-pane -t "$SESSION:$WINDOW.$PANE" -p -S -10 2>/dev/null)
    
    # Check if codex is currently Working
    if echo "$LAST_LINES" | grep -q "Working"; then
        echo "[$(date)] Codex is still Working. Good."
        continue
    fi
    
    # Check if we see the idle prompt (no Working indicator)
    if echo "$LAST_LINES" | grep -q "gpt-5.4 default"; then
        echo "[$(date)] Codex appears idle. Urging to continue..."
        # Send a nudge message and Enter
        tmux send-keys -t "$SESSION:$WINDOW.$PANE" "Continue working on the active SOXL plan. Do not stop. Pick the next highest-priority task, execute it, and log results." Enter
    else
        echo "[$(date)] Unclear state. Last lines:"
        echo "$LAST_LINES" | tail -3
    fi
done
