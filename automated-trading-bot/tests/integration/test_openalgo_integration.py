"""
Integration Tests for OpenAlgo Client
Tests communication with OpenAlgo API and WebSocket
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx
import websockets

from src.integrations.openalgo_client import OpenAlgoClient
from tests.conftest import MockWebSocket


class TestOpenAlgoIntegration:
    """Integration tests for OpenAlgo client"""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, test_config):
        """Test OpenAlgo client initialization"""
        client = OpenAlgoClient(test_config)
        
        # Verify configuration
        assert client.base_url == "http://127.0.0.1:5000/api/v1"
        assert client.ws_url == "ws://127.0.0.1:8765"
        assert client.api_key == ""  # Empty in test config
    
    @pytest.mark.asyncio
    async def test_rest_api_communication(self, test_config):
        """Test REST API endpoints"""
        client = OpenAlgoClient(test_config)
        
        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"available_balance": 500000}
        
        with patch.object(client.client, 'get', return_value=mock_response) as mock_get:
            funds = await client.get_funds()
            
            assert funds["available_balance"] == 500000
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_config):
        """Test WebSocket connection and reconnection"""
        client = OpenAlgoClient(test_config)
        
        # Mock WebSocket
        mock_ws = MockWebSocket()
        
        with patch('websockets.connect', return_value=mock_ws):
            # Connect
            connect_task = asyncio.create_task(client.connect_websocket())
            
            # Wait for connection
            await asyncio.sleep(0.1)
            
            # Verify connected
            assert mock_ws.is_connected
            
            # Disconnect
            await client.disconnect_websocket()
            
            # Cancel connect task
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_market_data_subscription(self, test_config):
        """Test market data subscription via WebSocket"""
        client = OpenAlgoClient(test_config)
        
        received_data = []
        
        # Set up market data handler
        async def on_data(symbol, data):
            received_data.append((symbol, data))
        
        client.on_market_data = on_data
        
        # Mock WebSocket with market data
        mock_ws = MockWebSocket()
        market_update = {
            "type": "market_data",
            "symbol": "NIFTY",
            "data": {
                "ltp": 20000,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with patch('websockets.connect', return_value=mock_ws):
            # Connect and subscribe
            connect_task = asyncio.create_task(client.connect_websocket())
            await asyncio.sleep(0.1)
            
            await client.subscribe_market_data(["NIFTY"])
            
            # Simulate receiving market data
            mock_ws.add_message(market_update)
            await client._handle_message(json.dumps(market_update))
            
            # Verify data received
            assert len(received_data) == 1
            assert received_data[0][0] == "NIFTY"
            assert received_data[0][1]["ltp"] == 20000
            
            # Cleanup
            await client.disconnect_websocket()
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_order_placement(self, test_config):
        """Test order placement functionality"""
        client = OpenAlgoClient(test_config)
        
        order_params = {
            "symbol": "NIFTY24FEB20000CE",
            "exchange": "NFO",
            "action": "SELL",
            "quantity": 50,
            "product": "MIS",
            "order_type": "MARKET",
            "price": 0
        }
        
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "orderid": "240123000012345",
            "message": "Order placed successfully"
        }
        
        with patch.object(client.client, 'post', return_value=mock_response) as mock_post:
            result = await client.place_order(**order_params)
            
            assert result["status"] == "success"
            assert result["orderid"] == "240123000012345"
            
            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/placeorder"
            assert call_args[1]["json"]["symbol"] == "NIFTY24FEB20000CE"
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, test_config):
        """Test order cancellation"""
        client = OpenAlgoClient(test_config)
        
        order_id = "240123000012345"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "message": "Order cancelled successfully"
        }
        
        with patch.object(client.client, 'post', return_value=mock_response) as mock_post:
            result = await client.cancel_order(order_id)
            
            assert result["status"] == "success"
            mock_post.assert_called_once_with(
                "/cancelorder",
                json={"orderid": order_id}
            )
    
    @pytest.mark.asyncio
    async def test_position_retrieval(self, test_config):
        """Test getting positions"""
        client = OpenAlgoClient(test_config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "symbol": "NIFTY24FEB20000CE",
                "quantity": -50,
                "product": "MIS",
                "pnl": 2500
            }
        ]
        
        with patch.object(client.client, 'get', return_value=mock_response) as mock_get:
            positions = await client.get_positions()
            
            assert len(positions) == 1
            assert positions[0]["symbol"] == "NIFTY24FEB20000CE"
            assert positions[0]["pnl"] == 2500
            
            mock_get.assert_called_once_with("/positions")
    
    @pytest.mark.asyncio
    async def test_option_chain_retrieval(self, test_config):
        """Test getting option chain data"""
        client = OpenAlgoClient(test_config)
        
        symbol = "NIFTY"
        expiry = "2024-02-29"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "strikes": {
                "20000": {
                    "CE": {"ltp": 150, "iv": 20, "oi": 50000},
                    "PE": {"ltp": 140, "iv": 19, "oi": 45000}
                }
            }
        }
        
        with patch.object(client.client, 'get', return_value=mock_response) as mock_get:
            option_chain = await client.get_option_chain(symbol, expiry)
            
            assert "20000" in option_chain["strikes"]
            assert option_chain["strikes"]["20000"]["CE"]["ltp"] == 150
            
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_config):
        """Test error handling for API failures"""
        client = OpenAlgoClient(test_config)
        
        # Test connection error
        with patch.object(client.client, 'get', side_effect=httpx.ConnectError("Connection failed")):
            funds = await client.get_funds()
            assert funds == {}  # Should return empty dict on error
        
        # Test API error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid request"}
        
        with patch.object(client.client, 'post', return_value=mock_response):
            result = await client.place_order(
                symbol="INVALID",
                exchange="NFO",
                action="BUY",
                quantity=50
            )
            assert result.get("status") != "success"
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self, test_config):
        """Test automatic WebSocket reconnection"""
        client = OpenAlgoClient(test_config)
        
        # Track connection attempts
        connection_count = 0
        
        async def mock_connect(*args, **kwargs):
            nonlocal connection_count
            connection_count += 1
            
            if connection_count == 1:
                # First connection fails
                raise websockets.exceptions.ConnectionClosed(None, None)
            else:
                # Second connection succeeds
                return MockWebSocket()
        
        with patch('websockets.connect', side_effect=mock_connect):
            connect_task = asyncio.create_task(client.connect_websocket())
            
            # Wait for reconnection
            await asyncio.sleep(2)
            
            # Should have attempted reconnection
            assert connection_count >= 2
            
            # Cleanup
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_config):
        """Test rate limiting handling"""
        client = OpenAlgoClient(test_config)
        
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429  # Too Many Requests
        mock_response.headers = {"Retry-After": "2"}
        
        call_count = 0
        
        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return mock_response
            else:
                # Second call succeeds
                success_response = Mock()
                success_response.status_code = 200
                success_response.json.return_value = {"available_balance": 500000}
                return success_response
        
        with patch.object(client.client, 'get', side_effect=mock_get):
            # Should retry after rate limit
            funds = await client.get_funds()
            
            # Should eventually succeed
            assert funds.get("available_balance") == 500000
            assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_bulk_order_placement(self, test_config):
        """Test placing multiple orders concurrently"""
        client = OpenAlgoClient(test_config)
        
        orders = [
            {
                "symbol": f"STOCK{i}",
                "exchange": "NSE",
                "action": "BUY",
                "quantity": 100,
                "product": "CNC",
                "order_type": "MARKET"
            }
            for i in range(10)
        ]
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "status": "success",
            "orderid": "12345"
        }
        
        with patch.object(client.client, 'post', return_value=success_response):
            # Place orders concurrently
            tasks = [client.place_order(**order) for order in orders]
            results = await asyncio.gather(*tasks)
            
            # All orders should succeed
            assert len(results) == 10
            assert all(r["status"] == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_market_data_streaming(self, test_config):
        """Test continuous market data streaming"""
        client = OpenAlgoClient(test_config)
        
        received_updates = []
        
        async def on_data(symbol, data):
            received_updates.append((symbol, data))
        
        client.on_market_data = on_data
        
        # Mock WebSocket with multiple updates
        mock_ws = MockWebSocket()
        
        # Simulate rapid market updates
        updates = [
            {
                "type": "market_data",
                "symbol": "NIFTY",
                "data": {
                    "ltp": 20000 + i,
                    "volume": 1000000 + i * 1000,
                    "timestamp": datetime.now().isoformat()
                }
            }
            for i in range(5)
        ]
        
        with patch('websockets.connect', return_value=mock_ws):
            connect_task = asyncio.create_task(client.connect_websocket())
            await asyncio.sleep(0.1)
            
            # Subscribe to market data
            await client.subscribe_market_data(["NIFTY"])
            
            # Send all updates
            for update in updates:
                await client._handle_message(json.dumps(update))
            
            # Verify all updates received
            assert len(received_updates) == 5
            
            # Verify data integrity
            for i, (symbol, data) in enumerate(received_updates):
                assert symbol == "NIFTY"
                assert data["ltp"] == 20000 + i
            
            # Cleanup
            await client.disconnect_websocket()
            connect_task.cancel()
            try:
                await connect_task
            except asyncio.CancelledError:
                pass