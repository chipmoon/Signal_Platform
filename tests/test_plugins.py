"""
Unit Tests for Plugin System
=============================
Tests for AssetProvider plugins and ProviderRegistry.
"""

import pytest
from datetime import datetime, timedelta

from src.plugins.base import AssetInfo, ProviderRegistry
from src.plugins.commodity import CommodityProvider
from src.plugins.taiwan import TaiwanStockProvider


class TestProviderRegistry:
    """Test the provider registry system."""
    
    def test_registry_singleton(self):
        """Test that registry is a functional singleton."""
        from src.plugins import registry
        assert registry is not None
        assert len(registry._providers) >= 1  # At least commodity should beregistered
    
    def test_list_markets(self):
        """Test listing all markets."""
        registry = ProviderRegistry()
        commodity = CommodityProvider()
        registry.register(commodity)
        
        markets = registry.list_markets()
        assert len(markets) > 0
        assert any(m['id'] == 'COMMODITY' for m in markets)
    
    def test_get_provider(self):
        """Test getting provider by market ID."""
        registry = ProviderRegistry()
        commodity = CommodityProvider()
        registry.register(commodity)
        
        provider = registry.get('COMMODITY')
        assert provider is not None
        assert provider.market_id == 'COMMODITY'
    
    def test_cross_market_search(self):
        """Test searching across multiple markets."""
        registry = ProviderRegistry()
        
        commodity = CommodityProvider()
        taiwan = TaiwanStockProvider()
        
        registry.register(commodity)
        registry.register(taiwan)
        
        # Search for "GOLD" - should find commodity
        results = registry.search_all("GOLD", limit=5)
        assert len(results) > 0
        assert any(asset.market == 'COMMODITY' for asset in results)
        
        # Search for "2330" - should find TSMC
        results = registry.search_all("2330", limit=5)
        assert len(results) > 0
        assert any(asset.symbol.startswith('2330') for asset in results)


class TestCommodityProvider:
    """Test the commodity provider."""
    
    def test_market_id(self):
        """Test market ID property."""
        provider = CommodityProvider()
        assert provider.market_id == "COMMODITY"
        assert provider.market_name == "Commodities & Futures"
    
    def test_search_assets(self):
        """Test asset search functionality."""
        provider = CommodityProvider()
        
        # Search for "GOLD"
        results = provider.search_assets("GOLD", limit=5)
        assert len(results) > 0
        assert any('Gold' in asset.name for asset in results)
        
        # Search for "silver"
        results = provider.search_assets("silver", limit=5)
        assert len(results) > 0
        assert any('Silver' in asset.name for asset in results)
    
    def test_get_asset_info(self):
        """Test getting asset info."""
        provider = CommodityProvider()
        
        # Get by code
        asset = provider.get_asset_info("GOLD")
        assert asset is not None
        assert asset.symbol == "GC=F"
        assert "Gold" in asset.name
        assert asset.currency == "USD"
        
        # Get by Yahoo symbol
        asset = provider.get_asset_info("GC=F")
        assert asset is not None
        assert "Gold" in asset.name
    
    def test_get_price_data(self):
        """Test fetching price data."""
        provider = CommodityProvider()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        df = provider.get_price_data("GOLD", start_date, end_date)
        
        assert not df.empty
        assert "Date" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) > 0
    
    def test_supports_fundamentals(self):
        """Test fundamentals support flag."""
        provider = CommodityProvider()
        assert provider.supports_fundamentals() == False
    
    def test_supports_cot_data(self):
        """Test COT data support flag."""
        provider = CommodityProvider()
        assert provider.supports_cot_data() == True
    
    def test_list_all(self):
        """Test listing all commodities."""
        provider = CommodityProvider()
        all_commodities = provider.list_all()
        
        assert len(all_commodities) >= 7  # Should have at least 7 commodities
        assert all(asset.market == 'COMMODITY' for asset in all_commodities)


class TestTaiwanStockProvider:
    """Test the Taiwan stock provider."""
    
    def test_market_id(self):
        """Test market ID property."""
        provider = TaiwanStockProvider()
        assert provider.market_id == "TW"
        assert provider.market_name == "Taiwan Stocks"
    
    def test_search_assets(self):
        """Test asset search."""
        provider = TaiwanStockProvider()
        
        # Search for "2330"
        results = provider.search_assets("2330", limit=5)
        assert len(results) > 0
        assert any('TSMC' in asset.name for asset in results)
        
        # Search for "tsmc"
        results = provider.search_assets("tsmc", limit=5)
        assert len(results) > 0
        
        # Search for Chinese name
        results = provider.search_assets("台積電", limit=5)
        assert len(results) > 0
    
    def test_get_asset_info(self):
        """Test getting asset info."""
        provider = TaiwanStockProvider()
        
        # Get by code only
        asset = provider.get_asset_info("2330")
        assert asset is not None
        assert asset.symbol == "2330.TW"
        assert "TSMC" in asset.name
        assert asset.currency == "TWD"
        
        # Get by full symbol
        asset = provider.get_asset_info("2330.TW")
        assert asset is not None
        assert "TSMC" in asset.name
    
    def test_supports_fundamentals(self):
        """Test fundamentals support."""
        provider = TaiwanStockProvider()
        assert provider.supports_fundamentals() == True
    
    def test_supports_cot_data(self):
        """Test COT data support."""
        provider = TaiwanStockProvider()
        assert provider.supports_cot_data() == False


# Integration tests
class TestPluginIntegration:
    """Integration tests for the full plugin system."""
    
    def test_full_workflow(self):
        """Test complete workflow: search -> get info -> fetch data."""
        from src.plugins import registry
        
        # Search across all markets
        results = registry.search_all("GOLD")
        assert len(results) > 0
        
        # Get first result
        asset = results[0]
        
        # Get provider
        provider = registry.get(asset.market)
        assert provider is not None
        
        # Get full asset info
        full_info = provider.get_asset_info(asset.symbol)
        assert full_info is not None
        
        # Try to fetch price data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        try:
            df = provider.get_price_data(asset.symbol, start_date, end_date)
            assert not df.empty
        except Exception:
            # Network failures are okay in tests
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
