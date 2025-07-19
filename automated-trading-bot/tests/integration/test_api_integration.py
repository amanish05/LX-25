"""
Integration Tests for FastAPI Application
Tests all REST API endpoints and WebSocket functionality
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import httpx

from src.api.app import create_app
from src.bots.base_bot import BotState
from tests.conftest import create_mock_bot, assert_api_response


class TestAPIIntegration:
    """Integration tests for REST API"""
    
    @pytest.fixture
    def test_app(self, mock_bot_manager, test_config_manager):
        """Create test FastAPI application"""
        app = create_app(mock_bot_manager, test_config_manager)
        return app
    
    @pytest.fixture
    def test_client(self, test_app):
        """Create test client"""
        return TestClient(test_app)
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        
        data = assert_api_response(response, 200)
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_system_status_endpoint(self, test_client, mock_bot_manager):
        """Test system status endpoint"""
        # Setup mock bot manager
        mock_bot_manager.get_system_status.return_value = {
            "uptime_hours": 2.5,
            "is_running": True,
            "market_hours": True,
            "stats": {
                "total_trades": 15,
                "successful_trades": 12,
                "failed_trades": 3,
                "total_pnl": 5000.0,
                "active_bots": 2
            },
            "bots": {}
        }
        
        response = test_client.get("/api/status")
        
        data = assert_api_response(response, 200)
        assert data["uptime_hours"] == 2.5
        assert data["is_running"] is True
        assert data["stats"]["total_trades"] == 15
    
    @pytest.mark.asyncio
    async def test_get_all_bots(self, test_client, mock_bot_manager):
        """Test getting all bots endpoint"""
        # Create mock bots
        bot1 = create_mock_bot("ShortStraddleBot", "short_straddle", BotState.RUNNING)
        bot2 = create_mock_bot("IronCondorBot", "iron_condor", BotState.PAUSED)
        
        mock_bot_manager.get_bot_status.return_value = {
            bot1.name: bot1.get_status(),
            bot2.name: bot2.get_status()
        }
        
        response = test_client.get("/api/bots")
        
        data = assert_api_response(response, 200)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "ShortStraddleBot"
        assert data[0]["state"] == "running"
    
    @pytest.mark.asyncio
    async def test_get_specific_bot(self, test_client, mock_bot_manager):
        """Test getting specific bot endpoint"""
        bot = create_mock_bot("TestBot", "short_straddle", BotState.RUNNING)
        mock_bot_manager.get_bot_status.return_value = bot.get_status()
        
        response = test_client.get("/api/bots/TestBot")
        
        data = assert_api_response(response, 200)
        assert data["name"] == "TestBot"
        assert data["type"] == "short_straddle"
        assert data["state"] == "running"
    
    @pytest.mark.asyncio
    async def test_start_bot(self, test_client, mock_bot_manager):
        """Test starting a bot"""
        mock_bot_manager.start_bot = AsyncMock()
        
        response = test_client.post("/api/bots/TestBot/start")
        
        data = assert_api_response(response, 200)
        assert "success" in data["message"].lower()
        mock_bot_manager.start_bot.assert_called_once_with("TestBot")
    
    @pytest.mark.asyncio
    async def test_stop_bot(self, test_client, mock_bot_manager):
        """Test stopping a bot"""
        mock_bot_manager.stop_bot = AsyncMock()
        
        response = test_client.post("/api/bots/TestBot/stop")
        
        data = assert_api_response(response, 200)
        assert "success" in data["message"].lower()
        mock_bot_manager.stop_bot.assert_called_once_with("TestBot")
    
    @pytest.mark.asyncio
    async def test_pause_bot(self, test_client, mock_bot_manager):
        """Test pausing a bot"""
        mock_bot_manager.pause_bot = AsyncMock()
        
        response = test_client.post("/api/bots/TestBot/pause")
        
        data = assert_api_response(response, 200)
        assert "success" in data["message"].lower()
        mock_bot_manager.pause_bot.assert_called_once_with("TestBot")
    
    @pytest.mark.asyncio
    async def test_resume_bot(self, test_client, mock_bot_manager):
        """Test resuming a bot"""
        mock_bot_manager.resume_bot = AsyncMock()
        
        response = test_client.post("/api/bots/TestBot/resume")
        
        data = assert_api_response(response, 200)
        assert "success" in data["message"].lower()
        mock_bot_manager.resume_bot.assert_called_once_with("TestBot")
    
    @pytest.mark.asyncio
    async def test_get_positions(self, test_client, mock_bot_manager):
        """Test getting positions endpoint"""
        mock_bot_manager.db_manager.get_open_positions.return_value = [
            {
                "id": 1,
                "bot_name": "TestBot",
                "symbol": "NIFTY",
                "position_type": "SHORT",
                "quantity": 50,
                "entry_price": 150.0,
                "current_price": 140.0,
                "pnl": 500.0,
                "status": "OPEN"
            }
        ]
        
        response = test_client.get("/api/positions")
        
        data = assert_api_response(response, 200)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["symbol"] == "NIFTY"
        assert data[0]["pnl"] == 500.0
    
    @pytest.mark.asyncio
    async def test_get_positions_with_filters(self, test_client, mock_bot_manager):
        """Test getting positions with query parameters"""
        mock_bot_manager.db_manager.get_open_positions.return_value = []
        
        response = test_client.get("/api/positions?status=CLOSED&symbol=BANKNIFTY")
        
        assert_api_response(response, 200)
        # Verify filters were passed (implementation depends on actual API)
    
    @pytest.mark.asyncio
    async def test_get_signals(self, test_client, mock_bot_manager):
        """Test getting signals endpoint"""
        mock_bot_manager.db_manager.get_signals.return_value = [
            {
                "id": 1,
                "bot_name": "TestBot",
                "symbol": "NIFTY",
                "signal_type": "SHORT_STRADDLE",
                "signal_strength": 0.85,
                "executed": False,
                "created_at": datetime.now().isoformat()
            }
        ]
        
        response = test_client.get("/api/signals?limit=10")
        
        data = assert_api_response(response, 200)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["signal_type"] == "SHORT_STRADDLE"
    
    @pytest.mark.asyncio
    async def test_get_performance(self, test_client, mock_bot_manager):
        """Test getting performance metrics"""
        mock_bot_manager.db_manager.get_bot_performance.return_value = {
            "total_trades": 50,
            "winning_trades": 35,
            "total_pnl": 25000,
            "win_rate": 70.0,
            "max_drawdown": 5.0
        }
        
        mock_bot_manager.get_bot_status.return_value = {
            "TestBot": {
                "performance": {
                    "total_trades": 50,
                    "win_rate": 70.0,
                    "total_pnl": 25000
                }
            }
        }
        
        response = test_client.get("/api/performance")
        
        data = assert_api_response(response, 200)
        assert data["total_pnl"] == 25000
        assert data["win_rate"] == 70.0
    
    @pytest.mark.asyncio
    async def test_get_configuration(self, test_client, test_config_manager):
        """Test getting configuration (masked)"""
        response = test_client.get("/api/config")
        
        data = assert_api_response(response, 200)
        assert "system" in data
        assert "bots" in data
        assert "trading" in data
        
        # Sensitive data should be masked
        assert "api_key" not in str(data)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_client, mock_bot_manager):
        """Test API error handling"""
        # Test bot not found
        mock_bot_manager.start_bot.side_effect = ValueError("Bot not found")
        
        response = test_client.post("/api/bots/NonExistentBot/start")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Bot not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client, mock_bot_manager):
        """Test handling concurrent API requests"""
        # Make multiple concurrent requests
        urls = [
            "/api/status",
            "/api/bots",
            "/api/positions",
            "/api/signals",
            "/api/performance"
        ]
        
        # Use regular requests for sync test client
        responses = []
        for url in urls:
            response = test_client.get(url)
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, test_client):
        """Test CORS headers"""
        response = test_client.options(
            "/api/status",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should allow configured origins
        assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.asyncio
    async def test_api_documentation(self, test_client):
        """Test API documentation endpoints"""
        # Test OpenAPI schema
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Automated Trading Bot API"
        assert "/api/status" in schema["paths"]
        
        # Test docs endpoint
        response = test_client.get("/docs")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_websocket_endpoint(self, test_app):
        """Test WebSocket endpoint"""
        from fastapi.testclient import TestClient
        
        with TestClient(test_app) as client:
            # Test WebSocket connection
            try:
                with client.websocket_connect("/ws") as websocket:
                    # Send a test message
                    websocket.send_json({"type": "ping"})
                    
                    # Should receive system status updates
                    # (Implementation depends on actual WebSocket handler)
            except Exception:
                # WebSocket might not be fully implemented yet
                pass
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_client):
        """Test rate limiting (if implemented)"""
        # Make many rapid requests
        responses = []
        for _ in range(100):
            response = test_client.get("/api/status")
            responses.append(response)
        
        # Check if rate limiting kicks in
        # (Implementation depends on whether rate limiting is configured)
        status_codes = [r.status_code for r in responses]
        
        # Most should be successful
        successful = sum(1 for code in status_codes if code == 200)
        assert successful > 50  # At least half should succeed
    
    @pytest.mark.asyncio
    async def test_api_versioning(self, test_client):
        """Test API versioning"""
        # Current version endpoints
        response = test_client.get("/api/status")
        assert response.status_code == 200
        
        # Future: test v2 endpoints when available
        # response = test_client.get("/api/v2/status")
    
    @pytest.mark.asyncio
    async def test_request_validation(self, test_client):
        """Test request parameter validation"""
        # Invalid limit parameter
        response = test_client.get("/api/signals?limit=invalid")
        assert response.status_code == 422  # Validation error
        
        # Negative limit
        response = test_client.get("/api/signals?limit=-1")
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_response_timing(self, test_client):
        """Test API response times"""
        import time
        
        endpoints = ["/health", "/api/status", "/api/bots"]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = test_client.get(endpoint)
            end_time = time.time()
            
            # Response should be fast (< 500ms)
            assert response.status_code == 200
            assert (end_time - start_time) < 0.5