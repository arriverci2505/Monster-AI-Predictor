#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT MANAGER v15.0                                               ║
║  Command-line utility for managing the trading bot                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python bot_manager.py start      # Start the bot
    python bot_manager.py stop       # Stop the bot
    python bot_manager.py restart    # Restart the bot
    python bot_manager.py status     # Check bot status
    python bot_manager.py backup     # Create backup
    python bot_manager.py stats      # Show statistics
"""

import os
import sys
import json
import time
import psutil
import subprocess
import shutil
from datetime import datetime

# Configuration
STATE_FILE = os.path.abspath("bot_state_v14_4.json")
BACKUP_DIR = "backups"
ENGINE_SCRIPT = "monster_engine.py"
UI_SCRIPT = "monster_ui.py"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print fancy header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          MONSTER BOT MANAGER v15.0                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

def is_bot_running():
    """Check if monster_engine.py is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and ENGINE_SCRIPT in ' '.join(proc.info['cmdline']):
                return True, proc.info['pid']
        except:
            pass
    return False, None

def is_ui_running():
    """Check if Streamlit UI is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and 'streamlit' in ' '.join(proc.info['cmdline']):
                return True, proc.info['pid']
        except:
            pass
    return False, None

def start_bot():
    """Start the trading bot"""
    print(f"{Colors.YELLOW}Starting Monster Engine...{Colors.END}")
    
    running, pid = is_bot_running()
    if running:
        print(f"{Colors.RED}✗ Bot is already running (PID: {pid}){Colors.END}")
        return
    
    # Start engine
    try:
        subprocess.Popen([sys.executable, ENGINE_SCRIPT], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        running, pid = is_bot_running()
        if running:
            print(f"{Colors.GREEN}✓ Bot started successfully (PID: {pid}){Colors.END}")
        else:
            print(f"{Colors.RED}✗ Failed to start bot{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def stop_bot():
    """Stop the trading bot"""
    print(f"{Colors.YELLOW}Stopping Monster Engine...{Colors.END}")
    
    running, pid = is_bot_running()
    if not running:
        print(f"{Colors.RED}✗ Bot is not running{Colors.END}")
        return
    
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        
        # Update state file
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            state['bot_status'] = 'Stopped'
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        
        print(f"{Colors.GREEN}✓ Bot stopped successfully{Colors.END}")
    except psutil.TimeoutExpired:
        print(f"{Colors.YELLOW}Forcing bot to stop...{Colors.END}")
        process.kill()
        print(f"{Colors.GREEN}✓ Bot force-stopped{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def restart_bot():
    """Restart the trading bot"""
    print(f"{Colors.YELLOW}Restarting Monster Engine...{Colors.END}")
    stop_bot()
    time.sleep(2)
    start_bot()

def show_status():
    """Show bot status"""
    print(f"{Colors.BOLD}System Status:{Colors.END}\n")
    
    # Bot status
    bot_running, bot_pid = is_bot_running()
    ui_running, ui_pid = is_ui_running()
    
    if bot_running:
        print(f"{Colors.GREEN}● Bot Engine: ONLINE{Colors.END} (PID: {bot_pid})")
    else:
        print(f"{Colors.RED}● Bot Engine: OFFLINE{Colors.END}")
    
    if ui_running:
        print(f"{Colors.GREEN}● Streamlit UI: ONLINE{Colors.END} (PID: {ui_pid})")
    else:
        print(f"{Colors.RED}● Streamlit UI: OFFLINE{Colors.END}")
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"\n{Colors.BOLD}System Resources:{Colors.END}")
    print(f"  CPU Usage: {cpu_percent:.1f}%")
    print(f"  RAM Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
    
    # State file info
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            print(f"\n{Colors.BOLD}Bot State:{Colors.END}")
            print(f"  Status: {state.get('bot_status', 'Unknown')}")
            print(f"  Balance: ${state.get('balance', 0):,.2f}")
            print(f"  Current Price: ${state.get('current_price', 0):,.2f}")
            print(f"  Open Trades: {len(state.get('open_trades', []))}")
            print(f"  Pending Orders: {len(state.get('pending_orders', []))}")
            print(f"  Total Trades: {len(state.get('trade_history', []))}")
            print(f"  Win Rate: {state.get('win_rate', 0):.1f}%")
            print(f"  Last Update: {state.get('last_update_time', 'N/A')}")
        except Exception as e:
            print(f"\n{Colors.RED}Error reading state: {e}{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠ State file not found{Colors.END}")

def create_backup():
    """Create backup of current state"""
    print(f"{Colors.YELLOW}Creating backup...{Colors.END}")
    
    if not os.path.exists(STATE_FILE):
        print(f"{Colors.RED}✗ State file not found{Colors.END}")
        return
    
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"bot_state_{timestamp}.json")
    
    try:
        shutil.copy(STATE_FILE, backup_file)
        file_size = os.path.getsize(backup_file)
        print(f"{Colors.GREEN}✓ Backup created: {backup_file}{Colors.END}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        
        # Show total backups
        backup_count = len([f for f in os.listdir(BACKUP_DIR) if f.endswith('.json')])
        print(f"  Total backups: {backup_count}")
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def show_stats():
    """Show detailed statistics"""
    if not os.path.exists(STATE_FILE):
        print(f"{Colors.RED}✗ State file not found{Colors.END}")
        return
    
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        
        history = state.get('trade_history', [])
        
        print(f"{Colors.BOLD}Trading Statistics:{Colors.END}\n")
        
        if not history:
            print(f"{Colors.YELLOW}No trades yet{Colors.END}")
            return
        
        # Calculate stats
        total_trades = len(history)
        wins = len([t for t in history if float(str(t.get('net_pnl', '0%')).replace('%', '')) > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate total PnL
        total_pnl = 0
        for trade in history:
            try:
                pnl_str = trade.get('dollar_pnl', '$0.00')
                pnl_value = float(pnl_str.replace('$', '').replace(',', ''))
                total_pnl += pnl_value
            except:
                pass
        
        # Calculate average PnL per trade
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {Colors.GREEN}{wins}{Colors.END}")
        print(f"Losses: {Colors.RED}{losses}{Colors.END}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"\nTotal PnL: ${total_pnl:,.2f}")
        print(f"Average PnL/Trade: ${avg_pnl:,.2f}")
        print(f"Current Balance: ${state.get('balance', 0):,.2f}")
        
        # Show last 5 trades
        print(f"\n{Colors.BOLD}Last 5 Trades:{Colors.END}")
        for i, trade in enumerate(history[:5], 1):
            pnl = trade.get('net_pnl', '0%')
            side = trade.get('side', 'N/A')
            exit_reason = trade.get('exit_reason', 'N/A')
            
            pnl_float = float(pnl.replace('%', ''))
            color = Colors.GREEN if pnl_float > 0 else Colors.RED
            
            print(f"  {i}. {side} | PnL: {color}{pnl}{Colors.END} | Exit: {exit_reason}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def start_ui():
    """Start Streamlit UI"""
    print(f"{Colors.YELLOW}Starting Streamlit UI...{Colors.END}")
    
    ui_running, _ = is_ui_running()
    if ui_running:
        print(f"{Colors.RED}✗ UI is already running{Colors.END}")
        return
    
    try:
        subprocess.Popen(['streamlit', 'run', UI_SCRIPT],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        time.sleep(2)
        print(f"{Colors.GREEN}✓ UI started successfully{Colors.END}")
        print(f"{Colors.CYAN}   Access at: http://localhost:8501{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")

def show_help():
    """Show help message"""
    print(f"{Colors.BOLD}Available Commands:{Colors.END}\n")
    print(f"  {Colors.CYAN}start{Colors.END}      - Start the trading bot")
    print(f"  {Colors.CYAN}stop{Colors.END}       - Stop the trading bot")
    print(f"  {Colors.CYAN}restart{Colors.END}    - Restart the trading bot")
    print(f"  {Colors.CYAN}status{Colors.END}     - Show bot status and system info")
    print(f"  {Colors.CYAN}stats{Colors.END}      - Show detailed trading statistics")
    print(f"  {Colors.CYAN}backup{Colors.END}     - Create backup of current state")
    print(f"  {Colors.CYAN}ui{Colors.END}         - Start Streamlit dashboard")
    print(f"  {Colors.CYAN}help{Colors.END}       - Show this help message")
    print()

def main():
    print_header()
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'start': start_bot,
        'stop': stop_bot,
        'restart': restart_bot,
        'status': show_status,
        'stats': show_stats,
        'backup': create_backup,
        'ui': start_ui,
        'help': show_help
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"{Colors.RED}✗ Unknown command: {command}{Colors.END}")
        print(f"{Colors.YELLOW}Use 'help' to see available commands{Colors.END}")

if __name__ == "__main__":
    main()
