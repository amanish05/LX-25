"""
FastAPI Application for Trading Bot System
Provides REST API endpoints for bot management and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ..core.bot_manager import BotManager
from ..config import ConfigManager
from ..utils.logger import TradingLogger
from .models import (
    BotStatusResponse, SystemStatusResponse, PositionResponse,
    SignalResponse, PerformanceResponse, BotActionRequest
)


def create_app(bot_manager: BotManager = None, config_manager: ConfigManager = None) -> FastAPI:
    """Create FastAPI application"""
    
    # Get API configuration
    api_config = config_manager.get_api_config()
    
    app = FastAPI(
        title=api_config['title'],
        description="REST API for managing and monitoring trading bots",
        version=api_config['version'],
        docs_url="/docs" if api_config['docs_enabled'] else None,
        redoc_url="/redoc" if api_config['docs_enabled'] else None
    )
    
    # Store bot manager and config in app state
    app.state.bot_manager = bot_manager
    app.state.config_manager = config_manager
    app.state.logger = TradingLogger(__name__)
    
    # Add CORS middleware
    cors_origins = api_config['cors_origins']
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    
    # System endpoints
    @app.get("/api/status", response_model=SystemStatusResponse)
    async def get_system_status():
        """Get overall system status"""
        try:
            if not app.state.bot_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Bot manager not initialized"
                )
            
            status = app.state.bot_manager.get_system_status()
            return SystemStatusResponse(**status)
            
        except Exception as e:
            app.state.logger.error(f"Error getting system status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Bot management endpoints
    @app.get("/api/bots", response_model=List[BotStatusResponse])
    async def get_all_bots():
        """Get status of all bots"""
        try:
            bot_statuses = app.state.bot_manager.get_bot_status()
            return [
                BotStatusResponse(**status) 
                for status in bot_statuses.values()
            ]
            
        except Exception as e:
            app.state.logger.error(f"Error getting bot statuses: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.get("/api/bots/{bot_name}", response_model=BotStatusResponse)
    async def get_bot_status(bot_name: str):
        """Get status of specific bot"""
        try:
            status = app.state.bot_manager.get_bot_status(bot_name)
            return BotStatusResponse(**status)
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error getting bot status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.post("/api/bots/{bot_name}/start")
    async def start_bot(bot_name: str):
        """Start a specific bot"""
        try:
            await app.state.bot_manager.start_bot(bot_name)
            return {"message": f"Bot {bot_name} started successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error starting bot: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.post("/api/bots/{bot_name}/stop")
    async def stop_bot(bot_name: str):
        """Stop a specific bot"""
        try:
            await app.state.bot_manager.stop_bot(bot_name)
            return {"message": f"Bot {bot_name} stopped successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error stopping bot: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.post("/api/bots/{bot_name}/pause")
    async def pause_bot(bot_name: str):
        """Pause a specific bot"""
        try:
            await app.state.bot_manager.pause_bot(bot_name)
            return {"message": f"Bot {bot_name} paused successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error pausing bot: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.post("/api/bots/{bot_name}/resume")
    async def resume_bot(bot_name: str):
        """Resume a specific bot"""
        try:
            await app.state.bot_manager.resume_bot(bot_name)
            return {"message": f"Bot {bot_name} resumed successfully"}
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error resuming bot: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Position endpoints
    @app.get("/api/positions", response_model=List[PositionResponse])
    async def get_all_positions():
        """Get all open positions across all bots"""
        try:
            positions = await app.state.bot_manager.db_manager.get_open_positions()
            return [PositionResponse(**pos) for pos in positions]
            
        except Exception as e:
            app.state.logger.error(f"Error getting positions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.get("/api/positions/{bot_name}", response_model=List[PositionResponse])
    async def get_bot_positions(bot_name: str):
        """Get positions for specific bot"""
        try:
            positions = await app.state.bot_manager.db_manager.get_open_positions(bot_name)
            return [PositionResponse(**pos) for pos in positions]
            
        except Exception as e:
            app.state.logger.error(f"Error getting bot positions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Signal endpoints
    @app.get("/api/signals", response_model=List[SignalResponse])
    async def get_recent_signals(limit: int = 50):
        """Get recent signals from all bots"""
        try:
            # This would need to be implemented in database manager
            signals = []  # Placeholder
            return [SignalResponse(**signal) for signal in signals]
            
        except Exception as e:
            app.state.logger.error(f"Error getting signals: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Performance endpoints
    @app.get("/api/performance", response_model=PerformanceResponse)
    async def get_system_performance():
        """Get overall system performance metrics"""
        try:
            # Aggregate performance from all bots
            performance = {
                "total_pnl": app.state.bot_manager.system_stats["total_pnl"],
                "total_trades": app.state.bot_manager.system_stats["total_trades"],
                "win_rate": 0,  # Calculate from bot performances
                "sharpe_ratio": 0,  # Calculate if needed
                "max_drawdown": 0,  # Calculate from bot performances
                "daily_pnl": {},  # Would need historical data
                "bot_performances": {}  # Individual bot performances
            }
            
            return PerformanceResponse(**performance)
            
        except Exception as e:
            app.state.logger.error(f"Error getting performance: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.get("/api/performance/{bot_name}", response_model=Dict[str, Any])
    async def get_bot_performance(bot_name: str):
        """Get performance metrics for specific bot"""
        try:
            bot_status = app.state.bot_manager.get_bot_status(bot_name)
            return bot_status.get("performance", {})
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            app.state.logger.error(f"Error getting bot performance: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # Configuration endpoints
    @app.get("/api/config")
    async def get_configuration():
        """Get current system configuration (sanitized)"""
        try:
            config_dict = app.state.config.to_dict()
            return config_dict
            
        except Exception as e:
            app.state.logger.error(f"Error getting configuration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # WebSocket endpoint for real-time updates
    from fastapi import WebSocket, WebSocketDisconnect
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates"""
        await websocket.accept()
        app.state.logger.info("WebSocket client connected")
        
        try:
            while True:
                # Send system status every 5 seconds
                status = app.state.bot_manager.get_system_status()
                await websocket.send_json({
                    "type": "system_status",
                    "data": status,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                await asyncio.sleep(5)
                
        except WebSocketDisconnect:
            app.state.logger.info("WebSocket client disconnected")
        except Exception as e:
            app.state.logger.error(f"WebSocket error: {e}")
            await websocket.close()
    
    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        app.state.logger.info("FastAPI application starting up")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        app.state.logger.info("FastAPI application shutting down")
    
    return app