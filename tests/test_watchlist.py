"""
Unit Tests for Watchlist Manager
=================================
Tests for watchlist persistence and management.
"""

import json
import os
import tempfile
import pytest

from src.watchlist import WatchlistItem, WatchlistManager


class TestWatchlistItem:
    """Test WatchlistItem dataclass."""
    
    def test_create_item(self):
        """Test creating a watchlist item."""
        item = WatchlistItem(
            symbol="GC=F",
            market="COMMODITY",
            name="Gold Futures",
            added_date="2026-02-16T12:00:00",
            notes="Test note"
        )
        
        assert item.symbol == "GC=F"
        assert item.market == "COMMODITY"
        assert item.name == "Gold Futures"
        assert item.notes == "Test note"
        assert item.alert_price is None


class TestWatchlistManager:
    """Test WatchlistManager functionality."""
    
    @pytest.fixture
    def temp_watchlist(self):
        """Create a temporary watchlist file."""
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    def test_create_manager(self, temp_watchlist):
        """Test creating a watchlist manager."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        assert len(manager.items) == 0
    
    def test_add_item(self, temp_watchlist):
        """Test adding an item to watchlist."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        result = manager.add(
            symbol="GC=F",
            market="COMMODITY",
            name="Gold Futures",
            notes="Watch for breakout"
        )
        
        assert result == True
        assert len(manager.items) == 1
        assert manager.contains("GC=F", "COMMODITY")
    
    def test_add_duplicate(self, temp_watchlist):
        """Test adding duplicate item."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold Futures")
        result = manager.add("GC=F", "COMMODITY", "Gold Futures")
        
        assert result == False  # Should reject duplicate
        assert len(manager.items) == 1
    
    def test_remove_item(self, temp_watchlist):
        """Test removing an item."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold Futures")
        assert len(manager.items) == 1
        
        result = manager.remove("GC=F", "COMMODITY")
        assert result == True
        assert len(manager.items) == 0
    
    def test_remove_nonexistent(self, temp_watchlist):
        """Test removing non-existent item."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        result = manager.remove("INVALID", "INVALID")
        assert result == False
    
    def test_persistence(self, temp_watchlist):
        """Test saving and loading."""
        # Create manager and add items
        manager1 = WatchlistManager(storage_path=temp_watchlist)
        manager1.add("GC=F", "COMMODITY", "Gold")
        manager1.add("2330.TW", "TW", "TSMC")
        assert len(manager1.items) == 2
        
        # Create new manager with same file
        manager2 = WatchlistManager(storage_path=temp_watchlist)
        assert len(manager2.items) == 2
        assert manager2.contains("GC=F", "COMMODITY")
        assert manager2.contains("2330.TW", "TW")
    
    def test_get_by_market(self, temp_watchlist):
        """Test filtering by market."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        manager.add("SI=F", "COMMODITY", "Silver")
        manager.add("2330.TW", "TW", "TSMC")
        
        commodity_items = manager.get_by_market("COMMODITY")
        assert len(commodity_items) == 2
        
        tw_items = manager.get_by_market("TW")
        assert len(tw_items) == 1
    
    def test_get_markets(self, temp_watchlist):
        """Test getting list of markets."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        manager.add("2330.TW", "TW", "TSMC")
        manager.add("VNM.VN", "VN", "Vinamilk")
        
        markets = manager.get_markets()
        assert len(markets) == 3
        assert "COMMODITY" in markets
        assert "TW" in markets
        assert "VN" in markets
    
    def test_update_notes(self, temp_watchlist):
        """Test updating notes."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        result = manager.update_notes("GC=F", "COMMODITY", "New note here")
        
        assert result == True
        item = manager.get("GC=F", "COMMODITY")
        assert item.notes == "New note here"
    
    def test_set_alert(self, temp_watchlist):
        """Test setting price alert."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        result = manager.set_alert("GC=F", "COMMODITY", 2100.0)
        
        assert result == True
        item = manager.get("GC=F", "COMMODITY")
        assert item.alert_price == 2100.0
    
    def test_export_symbols(self, temp_watchlist):
        """Test exporting symbols."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        manager.add("2330.TW", "TW", "TSMC")
        
        # Export all
        symbols = manager.export_symbols()
        assert len(symbols) == 2
        assert "GC=F" in symbols
        assert "2330.TW" in symbols
        
        # Export by market
        commodity_symbols = manager.export_symbols(market="COMMODITY")
        assert len(commodity_symbols) == 1
        assert "GC=F" in commodity_symbols
    
    def test_clear(self, temp_watchlist):
        """Test clearing watchlist."""
        manager = WatchlistManager(storage_path=temp_watchlist)
        
        manager.add("GC=F", "COMMODITY", "Gold")
        manager.add("2330.TW", "TW", "TSMC")
        assert len(manager.items) == 2
        
        manager.clear()
        assert len(manager.items) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
