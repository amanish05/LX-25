"""
Main entry point for the Automated Trading Bot System
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path
import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.bot_manager import BotManager
from src.api.app import create_app
from src.config import get_config_manager
from src.utils.logger import setup_logging


console = Console()


class TradingBotSystem:
    """Main trading bot system orchestrator"""
    
    def __init__(self, config_dir: str = "config", dev_mode: bool = False):
        self.config_manager = get_config_manager(config_dir)
        self.dev_mode = dev_mode
        self.bot_manager = None
        self.api_server = None
        self.shutdown_event = None  # Will be created in async context
        
        # Get logging configuration
        logging_config = self.config_manager.get_logging_config()
        
        # Setup logging
        self.logger = setup_logging(
            log_level=logging_config['level'],
            log_file=logging_config['file']
        )
        
    async def start(self):
        """Start the trading bot system"""
        try:
            console.print("[bold green]Starting Automated Trading Bot System...[/bold green]")
            
            # Create shutdown event in async context
            self.shutdown_event = asyncio.Event()
            
            # Initialize bot manager
            self.bot_manager = BotManager(self.config_manager)
            
            # Create FastAPI app
            app = create_app(self.bot_manager, self.config_manager)
            
            # Get API configuration
            api_config_data = self.config_manager.get_api_config()
            
            # Configure API server
            api_config = uvicorn.Config(
                app,
                host=api_config_data['host'],
                port=api_config_data['port'],
                log_level=self.config_manager.app_config.logging.level.lower(),
                reload=self.dev_mode
            )
            self.api_server = uvicorn.Server(api_config)
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start components concurrently
            await asyncio.gather(
                self.bot_manager.start(),
                self.api_server.serve(),
                self._monitor_system()
            )
            
        except Exception as e:
            console.print(f"[bold red]Failed to start system: {e}[/bold red]")
            raise
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            console.print(f"\n[yellow]Received signal {signum}. Shutting down gracefully...[/yellow]")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _monitor_system(self):
        """Monitor system health and handle shutdown"""
        await self.shutdown_event.wait()
        
        console.print("[yellow]Initiating shutdown sequence...[/yellow]")
        
        # Stop bot manager
        if self.bot_manager:
            await self.bot_manager.stop()
        
        # Stop API server
        if self.api_server:
            self.api_server.should_exit = True
        
        console.print("[green]Shutdown complete[/green]")


@click.command()
@click.option('--config-dir', '-c', default='config', 
              help='Path to configuration directory')
@click.option('--dev', is_flag=True, 
              help='Run in development mode with auto-reload')
@click.option('--init-db', is_flag=True, 
              help='Initialize database tables')
@click.option('--show-config', is_flag=True,
              help='Show current configuration summary')
@click.version_option(version='1.0.0')
def main(config_dir: str, dev: bool, init_db: bool, show_config: bool):
    """Automated Trading Bot System for OpenAlgo"""
    
    config_manager = get_config_manager(config_dir)
    
    if show_config:
        # Show configuration summary
        console.print(config_manager.get_summary())
        return
    
    if init_db:
        # Initialize database
        from src.core.database import init_database
        asyncio.run(init_database(config_manager))
        console.print("[green]Database initialized successfully[/green]")
        return
    
    # ASCII art banner
    console.print("""
    [bold cyan]
    ╔═══════════════════════════════════════════════════════╗
    ║         Automated Trading Bot System v1.0.0           ║
    ║              Powered by OpenAlgo                      ║
    ╚═══════════════════════════════════════════════════════╝
    [/bold cyan]
    """)
    
    # Start the system
    system = TradingBotSystem(config_dir, dev)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]System error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()