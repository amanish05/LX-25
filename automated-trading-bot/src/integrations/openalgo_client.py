"""
OpenAlgo API Client Integration
Handles all communication with OpenAlgo REST API and WebSocket
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import httpx
import websockets
from websockets.exceptions import WebSocketException
import pandas as pd
from ..utils.logger import TradingLogger


class OpenAlgoClient:
    """Async client for OpenAlgo API integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get required environment variables - fail if missing
        self.api_url = self._get_required_env("OPENALGO_API_URL")
        self.api_key = self._get_required_env("OPENALGO_API_KEY")
        self.ws_url = self._get_required_env("OPENALGO_WEBSOCKET_URL")
        self.host = self._get_required_env("OPENALGO_HOST")
        
        # HTTP client configuration
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        
        self.retry_attempts = 3
        
        # WebSocket management
        self.ws_connection = None
        self.ws_callbacks: Dict[str, List[Callable]] = {}
        self.ws_reconnect_delay = 5
        self.ws_max_reconnect_attempts = 10
        
        self.logger = TradingLogger(__name__)
        
        # Create HTTP client
        self.http_client = httpx.AsyncClient(
            base_url=self.api_url,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )
    
    def _get_required_env(self, env_var: str) -> str:
        """Get required environment variable or raise error with clear message"""
        value = os.getenv(env_var)
        if not value:
            raise ValueError(
                f"Required environment variable '{env_var}' is not set. "
                f"Please check your .env file and ensure all OpenAlgo configuration is properly set."
            )
        return value
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close all connections"""
        if self.http_client:
            await self.http_client.aclose()
        
        if self.ws_connection:
            await self.ws_connection.close()
    
    # REST API Methods
    
    async def _request(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP POST request with retry logic"""
        last_exception = None
        
        # All OpenAlgo API calls are POST requests and require apikey in body
        request_data = data or {}
        request_data["apikey"] = self.api_key
        
        for attempt in range(self.retry_attempts):
            try:
                response = await self.http_client.post(endpoint, json=request_data)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 30)
                    self.logger.warning(f"Rate limited, waiting {wait_time}s", 
                                      endpoint=endpoint, attempt=attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = min(2 ** attempt, 10)
                    self.logger.warning(f"Request failed, retrying in {wait_time}s", 
                                      endpoint=endpoint, error=str(e), attempt=attempt)
                    await asyncio.sleep(wait_time)
        
        raise last_exception or Exception("Request failed")
    
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        return await self._request(
            "/quotes/",
            {"symbol": symbol, "exchange": exchange}
        )
    
    async def get_depth(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Get market depth for a symbol"""
        return await self._request(
            "/depth/",
            {"symbol": symbol, "exchange": exchange}
        )
    
    async def get_history(self, symbol: str, exchange: str, interval: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data"""
        data = await self._request(
            "/history/",
            {
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        # Convert to DataFrame
        if data and "data" in data:
            df = pd.DataFrame(data["data"])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        
        return pd.DataFrame()
    
    async def place_order(self, symbol: str, exchange: str, action: str,
                         quantity: int, product: str, price_type: str,
                         price: float = 0, trigger_price: float = 0,
                         disclosed_quantity: int = 0) -> Dict[str, Any]:
        """Place an order"""
        order_data = {
            "symbol": symbol,
            "exchange": exchange,
            "action": action,
            "quantity": quantity,
            "product": product,
            "pricetype": price_type,
            "price": price,
            "trigger_price": trigger_price,
            "disclosed_quantity": disclosed_quantity
        }
        
        response = await self._request("/placeorder/", order_data)
        
        self.logger.trade_executed(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price if price > 0 else "MARKET",
            order_id=response.get("orderid", "N/A")
        )
        
        return response
    
    async def place_smart_order(self, strategy: str, symbol: str, exchange: str,
                               action: str, quantity: int, position_size: int,
                               product: str, price_type: str, price: float = 0) -> Dict[str, Any]:
        """Place a smart order with strategy tracking"""
        order_data = {
            "strategy": strategy,
            "symbol": symbol,
            "exchange": exchange,
            "action": action,
            "quantity": quantity,
            "position_size": position_size,
            "product": product,
            "pricetype": price_type,
            "price": price
        }
        
        return await self._request("/placesmartorder/", order_data)
    
    async def modify_order(self, order_id: str, quantity: int = None,
                          price: float = None, trigger_price: float = None,
                          order_type: str = None) -> Dict[str, Any]:
        """Modify an existing order"""
        modify_data = {"orderid": order_id}
        
        if quantity is not None:
            modify_data["quantity"] = quantity
        if price is not None:
            modify_data["price"] = price
        if trigger_price is not None:
            modify_data["trigger_price"] = trigger_price
        if order_type is not None:
            modify_data["pricetype"] = order_type
        
        return await self._request("/modifyorder/", modify_data)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._request("/cancelorder/", {"orderid": order_id})
    
    async def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all pending orders"""
        return await self._request("/cancelallorder/")
    
    async def get_order_book(self) -> List[Dict[str, Any]]:
        """Get order book"""
        response = await self._request("/orderbook/")
        return response.get("data", [])
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of specific order"""
        return await self._request("/orderstatus/", {"orderid": order_id})
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        response = await self._request("/positionbook/")
        return response.get("data", [])
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings"""
        response = await self._request("/holdings/")
        return response.get("data", [])
    
    async def get_funds(self) -> Dict[str, Any]:
        """Get fund details"""
        return await self._request("/funds/")
    
    async def get_option_chain(self, symbol: str, expiry: str) -> Dict[str, Any]:
        """Get option chain data"""
        return await self._request(
            "/optionchain/",
            {"symbol": symbol, "expiry": expiry}
        )
    
    async def close_position(self, symbol: str, exchange: str, product: str,
                           position_size: int) -> Dict[str, Any]:
        """Close a position"""
        return await self._request(
            "/closeposition/",
            {
                "symbol": symbol,
                "exchange": exchange,
                "product": product,
                "position_size": position_size
            }
        )
    
    # WebSocket Methods
    
    async def connect_websocket(self):
        """Connect to WebSocket for live data"""
        self.logger.info("Connecting to WebSocket", url=self.ws_url)
        
        try:
            # For WebSocket, include API key in the URL or handle authentication after connection
            ws_url_with_auth = f"{self.ws_url}?apikey={self.api_key}"
            self.ws_connection = await websockets.connect(
                ws_url_with_auth,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.logger.info("WebSocket connected successfully")
            
            # Start listening for messages
            asyncio.create_task(self._ws_message_handler())
            
        except Exception as e:
            self.logger.error("WebSocket connection failed", error=str(e))
            raise
    
    async def _ws_message_handler(self):
        """Handle incoming WebSocket messages"""
        reconnect_attempts = 0
        
        while reconnect_attempts < self.ws_max_reconnect_attempts:
            try:
                if not self.ws_connection:
                    await self.connect_websocket()
                
                async for message in self.ws_connection:
                    try:
                        data = json.loads(message)
                        await self._process_ws_message(data)
                        reconnect_attempts = 0  # Reset on successful message
                        
                    except json.JSONDecodeError:
                        self.logger.error("Invalid WebSocket message", message=message)
                        
            except WebSocketException as e:
                self.logger.error("WebSocket error", error=str(e))
                reconnect_attempts += 1
                
                if reconnect_attempts < self.ws_max_reconnect_attempts:
                    wait_time = min(self.ws_reconnect_delay * reconnect_attempts, 60)
                    self.logger.info(f"Reconnecting WebSocket in {wait_time}s", 
                                   attempt=reconnect_attempts)
                    await asyncio.sleep(wait_time)
                    self.ws_connection = None
                else:
                    self.logger.error("Max WebSocket reconnection attempts reached")
                    break
    
    async def _process_ws_message(self, data: Dict[str, Any]):
        """Process WebSocket message and distribute to callbacks"""
        msg_type = data.get("type", "")
        symbol = data.get("symbol", "")
        
        # Get callbacks for this message type
        callbacks = self.ws_callbacks.get(msg_type, [])
        
        # Get callbacks for specific symbol
        symbol_callbacks = self.ws_callbacks.get(f"{msg_type}:{symbol}", [])
        
        # Execute all callbacks
        for callback in callbacks + symbol_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, callback, data)
            except Exception as e:
                self.logger.error("WebSocket callback error", error=str(e))
    
    async def subscribe_market_data(self, symbols: List[str], feed_type: str = "ltp",
                                   callback: Callable = None):
        """Subscribe to market data via WebSocket"""
        if not self.ws_connection:
            await self.connect_websocket()
        
        # Register callback if provided
        if callback:
            callback_key = f"{feed_type}"
            if callback_key not in self.ws_callbacks:
                self.ws_callbacks[callback_key] = []
            self.ws_callbacks[callback_key].append(callback)
        
        # Send subscription request
        sub_request = {
            "action": "subscribe",
            "symbols": symbols,
            "feed_type": feed_type
        }
        
        await self.ws_connection.send(json.dumps(sub_request))
        self.logger.info("Subscribed to market data", symbols=symbols, feed_type=feed_type)
    
    async def unsubscribe_market_data(self, symbols: List[str], feed_type: str = "ltp"):
        """Unsubscribe from market data"""
        if not self.ws_connection:
            return
        
        unsub_request = {
            "action": "unsubscribe",
            "symbols": symbols,
            "feed_type": feed_type
        }
        
        await self.ws_connection.send(json.dumps(unsub_request))
        self.logger.info("Unsubscribed from market data", symbols=symbols)
    
    async def disconnect_websocket(self):
        """Disconnect WebSocket"""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
            self.logger.info("WebSocket disconnected")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Get historical market data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1min, 5min, 15min, 1hour, 1day)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of candle data
        """
        try:
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            }
            
            response = await self._request("/history/", params)
            return response.get("data", [])
                    
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def get_historical_option_chain(
        self,
        symbol: str,
        expiry: str,
        date: str
    ) -> Dict[str, Any]:
        """Get historical option chain data
        
        Args:
            symbol: Underlying symbol
            expiry: Option expiry date
            date: Date for which to get the chain
            
        Returns:
            Option chain data
        """
        try:
            params = {
                "symbol": symbol,
                "expiry": expiry,
                "date": date
            }
            
            response = await self._request("/history/", params)
            return response
                    
        except Exception as e:
            self.logger.error(f"Error fetching historical option chain: {e}")
            return {}