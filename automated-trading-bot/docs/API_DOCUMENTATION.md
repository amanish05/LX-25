# Trading Bot System API Documentation

## Overview

The Automated Trading Bot System provides a REST API for managing and monitoring multiple trading bots. The API runs on port 8080 by default and includes WebSocket support for real-time updates.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API does not require authentication for local development. In production, you should implement proper authentication using API keys or JWT tokens.

## Endpoints

### Health Check

#### GET /health
Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

### System Status

#### GET /api/status
Get overall system status including all bots.

**Response:**
```json
{
  "uptime_hours": 2.5,
  "is_running": true,
  "market_hours": true,
  "stats": {
    "total_trades": 15,
    "successful_trades": 12,
    "failed_trades": 3,
    "total_pnl": 5000.0,
    "active_bots": 4
  },
  "bots": {
    "ShortStraddleBot": {
      "name": "ShortStraddleBot",
      "state": "running",
      "capital": {
        "initial": 200000,
        "current": 205000,
        "available": 150000,
        "locked": 55000
      },
      "positions": 2,
      "performance": {
        "total_trades": 5,
        "win_rate": 80.0,
        "total_pnl": 5000.0
      }
    }
  }
}
```

### Bot Management

#### GET /api/bots
Get status of all bots.

**Response:**
```json
[
  {
    "name": "ShortStraddleBot",
    "type": "short_straddle",
    "state": "running",
    "enabled": true,
    "capital": {
      "initial": 200000,
      "current": 205000,
      "available": 150000,
      "locked": 55000
    },
    "positions": 2,
    "open_orders": 0,
    "performance": {
      "total_trades": 5,
      "win_rate": 80.0,
      "total_pnl": 5000.0,
      "max_drawdown": 2.5
    },
    "last_health_check": "2024-01-15T10:25:00Z",
    "error_count": 0
  }
]
```

#### GET /api/bots/{bot_name}
Get status of a specific bot.

**Parameters:**
- `bot_name` (string, required): Name of the bot

**Response:** Same as single bot object above

#### POST /api/bots/{bot_name}/start
Start a specific bot.

**Parameters:**
- `bot_name` (string, required): Name of the bot

**Response:**
```json
{
  "message": "Bot ShortStraddleBot started successfully"
}
```

#### POST /api/bots/{bot_name}/stop
Stop a specific bot.

**Response:**
```json
{
  "message": "Bot ShortStraddleBot stopped successfully"
}
```

#### POST /api/bots/{bot_name}/pause
Pause a specific bot (stops taking new positions).

**Response:**
```json
{
  "message": "Bot ShortStraddleBot paused successfully"
}
```

#### POST /api/bots/{bot_name}/resume
Resume a paused bot.

**Response:**
```json
{
  "message": "Bot ShortStraddleBot resumed successfully"
}
```

### Positions

#### GET /api/positions
Get all open positions across all bots.

**Query Parameters:**
- `status` (string, optional): Filter by status (OPEN, CLOSED)
- `symbol` (string, optional): Filter by symbol

**Response:**
```json
[
  {
    "id": 1,
    "bot_name": "ShortStraddleBot",
    "symbol": "NIFTY",
    "exchange": "NFO",
    "position_type": "SHORT_STRADDLE",
    "quantity": 50,
    "entry_price": 290.5,
    "current_price": 285.0,
    "entry_time": "2024-01-15T09:30:00Z",
    "pnl": 275.0,
    "status": "OPEN"
  }
]
```

#### GET /api/positions/{bot_name}
Get positions for a specific bot.

### Signals

#### GET /api/signals
Get recent trading signals.

**Query Parameters:**
- `limit` (integer, optional): Number of signals to return (default: 50)
- `bot_name` (string, optional): Filter by bot
- `executed` (boolean, optional): Filter by execution status

**Response:**
```json
[
  {
    "id": 1,
    "bot_name": "ShortStraddleBot",
    "symbol": "NIFTY",
    "exchange": "NSE",
    "signal_type": "SHORT_STRADDLE",
    "signal_strength": 0.85,
    "executed": true,
    "created_at": "2024-01-15T09:25:00Z",
    "metadata": {
      "iv_rank": 82,
      "strike": 20000,
      "expiry": "2024-01-25"
    }
  }
]
```

### Performance

#### GET /api/performance
Get overall system performance metrics.

**Response:**
```json
{
  "total_pnl": 15000.0,
  "total_trades": 50,
  "win_rate": 70.0,
  "sharpe_ratio": 1.5,
  "max_drawdown": 5.0,
  "daily_pnl": {
    "2024-01-15": 2000.0,
    "2024-01-14": 1500.0,
    "2024-01-13": -500.0
  },
  "bot_performances": {
    "ShortStraddleBot": {
      "total_pnl": 5000.0,
      "win_rate": 80.0,
      "trades": 15
    }
  }
}
```

#### GET /api/performance/{bot_name}
Get performance metrics for a specific bot.

### Configuration

#### GET /api/config
Get current system configuration (sensitive data masked).

**Response:**
```json
{
  "system": {
    "environment": "development",
    "total_capital": 1000000
  },
  "bots": {
    "short_straddle": {
      "enabled": true,
      "capital": 200000,
      "max_positions": 2
    }
  },
  "trading": {
    "market_hours": {
      "start": "09:15",
      "end": "15:30",
      "timezone": "Asia/Kolkata"
    }
  }
}
```

## WebSocket

### WS /ws
Connect to WebSocket for real-time updates.

**Connection URL:**
```
ws://localhost:8080/ws
```

**Message Format:**
```json
{
  "type": "system_status",
  "data": {
    // System status object
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Message Types:**
- `system_status`: Overall system status
- `bot_update`: Individual bot status update
- `position_update`: Position changes
- `signal`: New trading signal
- `trade_execution`: Trade execution notification

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Bot not found",
  "detail": "Bot with name 'UnknownBot' does not exist",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

Currently no rate limiting is implemented. In production, consider adding rate limiting to prevent abuse.

## Examples

### Start a bot using curl:
```bash
curl -X POST http://localhost:8080/api/bots/ShortStraddleBot/start
```

### Get all positions:
```bash
curl http://localhost:8080/api/positions
```

### Connect to WebSocket using Python:
```python
import asyncio
import websockets
import json

async def listen():
    async with websockets.connect("ws://localhost:8080/ws") as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data['type']}")

asyncio.run(listen())
```

## SDK Usage

You can also use the Python SDK to interact with the API:

```python
from automated_trading_bot.client import TradingBotClient

client = TradingBotClient("http://localhost:8080")

# Get system status
status = client.get_status()

# Start a bot
client.start_bot("ShortStraddleBot")

# Get positions
positions = client.get_positions()
```