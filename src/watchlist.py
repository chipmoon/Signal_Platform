"""
Watchlist Manager
=================
Manages user's watchlist of assets across different markets.

Features:
- Add/remove assets
- Save/load from JSON
- Filter by market
- Add notes to assets
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class WatchlistItem:
    """Single watchlist entry."""
    
    symbol: str                      # e.g., "VNM.VN", "2330.TW", "GC=F"
    market: str                      # "VN", "TW", "COMMODITY"
    name: str                        # Display name
    added_date: str                  # ISO format
    notes: str = ""                  # User notes
    alert_price: Optional[float] = None  # Price alert level


class WatchlistManager:
    """
    Manages user's watchlist with persistence.
    
    Features:
    - Load/save from JSON file
    - Add/remove assets
    - Search and filter
    - Generate unique IDs for each asset
    """
    
    def __init__(self, storage_path: str = "watchlist.json"):
        """
        Initialize watchlist manager.
        
        Args:
            storage_path: Path to JSON file for persistence
        """
        self.storage_path = Path(storage_path)
        self.items: List[WatchlistItem] = []
        self._load()
    
    def _load(self) -> None:
        """Load watchlist from JSON file."""
        if not self.storage_path.exists():
            logger.info("No existing watchlist found, starting fresh")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.items = [
                WatchlistItem(**item)
                for item in data.get('items', [])
            ]
            
            logger.success(f"Loaded {len(self.items)} items from watchlist")
            
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            self.items = []
    
    def save(self) -> bool:
        """
        Save watchlist to JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create parent directory if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'version': '1.0',
                'updated': datetime.now().isoformat(),
                'items': [asdict(item) for item in self.items]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.items)} items to watchlist")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save watchlist: {e}")
            return False
    
    def add(
        self,
        symbol: str,
        market: str,
        name: str,
        notes: str = "",
        alert_price: Optional[float] = None
    ) -> bool:
        """
        Add asset to watchlist.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            name: Display name
            notes: Optional notes
            alert_price: Optional price alert level
            
        Returns:
            True if added, False if already exists
        """
        # Check if already in watchlist
        if self.contains(symbol, market):
            logger.warning(f"{symbol} ({market}) already in watchlist")
            return False
        
        item = WatchlistItem(
            symbol=symbol,
            market=market,
            name=name,
            added_date=datetime.now().isoformat(),
            notes=notes,
            alert_price=alert_price
        )
        
        self.items.append(item)
        self.save()
        
        logger.success(f"Added {symbol} ({market}) to watchlist")
        return True
    
    def remove(self, symbol: str, market: str) -> bool:
        """
        Remove asset from watchlist.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            
        Returns:
            True if removed, False if not found
        """
        original_count = len(self.items)
        self.items = [
            item for item in self.items
            if not (item.symbol == symbol and item.market == market)
        ]
        
        if len(self.items) < original_count:
            self.save()
            logger.success(f"Removed {symbol} ({market}) from watchlist")
            return True
        else:
            logger.warning(f"{symbol} ({market}) not found in watchlist")
            return False
    
    def contains(self, symbol: str, market: str) -> bool:
        """
        Check if asset is in watchlist.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            
        Returns:
            True if in watchlist
        """
        return any(
            item.symbol == symbol and item.market == market
            for item in self.items
        )
    
    def get(self, symbol: str, market: str) -> Optional[WatchlistItem]:
        """
        Get watchlist item by symbol and market.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            
        Returns:
            WatchlistItem if found, None otherwise
        """
        for item in self.items:
            if item.symbol == symbol and item.market == market:
                return item
        return None
    
    def update_notes(self, symbol: str, market: str, notes: str) -> bool:
        """
        Update notes for an asset.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            notes: New notes
            
        Returns:
            True if updated, False if not found
        """
        item = self.get(symbol, market)
        if item:
            item.notes = notes
            self.save()
            return True
        return False
    
    def set_alert(self, symbol: str, market: str, price: float) -> bool:
        """
        Set price alert for an asset.
        
        Args:
            symbol: Asset symbol
            market: Market ID
            price: Alert price level
            
        Returns:
            True if set, False if not found
        """
        item = self.get(symbol, market)
        if item:
            item.alert_price = price
            self.save()
            return True
        return False
    
    def get_all(self) -> List[WatchlistItem]:
        """Get all watchlist items."""
        return self.items.copy()
    
    def get_by_market(self, market: str) -> List[WatchlistItem]:
        """
        Get all items from a specific market.
        
        Args:
            market: Market ID
            
        Returns:
            List of watchlist items
        """
        return [item for item in self.items if item.market == market]
    
    def get_markets(self) -> List[str]:
        """
        Get list of unique markets in watchlist.
        
        Returns:
            List of market IDs
        """
        return list(set(item.market for item in self.items))
    
    def clear(self) -> None:
        """Clear all items from watchlist."""
        self.items = []
        self.save()
        logger.info("Cleared watchlist")
    
    def export_symbols(self, market: Optional[str] = None) -> List[str]:
        """
        Export list of symbols for backtesting/analysis.
        
        Args:
            market: Optional market filter
            
        Returns:
            List of symbols
        """
        if market:
            items = self.get_by_market(market)
        else:
            items = self.items
        
        return [item.symbol for item in items]
    
    def to_dict(self) -> Dict:
        """
        Export watchlist as dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'count': len(self.items),
            'markets': self.get_markets(),
            'items': [asdict(item) for item in self.items]
        }
