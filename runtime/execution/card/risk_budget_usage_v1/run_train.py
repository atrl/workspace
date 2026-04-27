#!/usr/bin/env python3
import sys
import os
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace')
os.chdir('/Users/rziiiii/.openclaw/workspace')

if __name__ == '__main__':
    sys.argv = ['cli', '--profiles', 'risk_budget_usage_v1', '--seed', '20260411', '--skip-data-refresh']
    from src.soxl_quant.app.cli import main
    main()
