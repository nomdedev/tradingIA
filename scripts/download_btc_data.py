"""
BTC Data Download Script
Downloads historical BTC/USD data from Alpaca API and saves to local files
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from api.data_fetcher import DataFetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DataFetcher not available: {e}")
    DATA_FETCHER_AVAILABLE = False
    DataFetcher = None

def download_btc_data(start_date, end_date, timeframe, output_dir='data/raw'):
    """
    Download BTC/USD data and save to CSV file

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Timeframe ('5Min', '15Min', '1Hour', '4Hour')
        output_dir: Output directory
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Map timeframe to filename
        timeframe_map = {
            '5Min': '5m',
            '15Min': '15m',
            '1Hour': '1h',
            '4Hour': '4h'
        }

        filename_tf = timeframe_map.get(timeframe, timeframe.lower().replace('min', 'm').replace('hour', 'h'))
        output_file = Path(output_dir) / f"btc_usd_{filename_tf}.csv"

        print(f"Downloading BTC/USD {timeframe} data from {start_date} to {end_date}")
        print(f"Output file: {output_file}")

        if DATA_FETCHER_AVAILABLE and DataFetcher is not None:
            # Use DataFetcher for download
            fetcher = DataFetcher()
            df = fetcher.get_historical_data(
                symbol='BTC/USD',
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                # Save to CSV
                df.to_csv(output_file)
                print(f"✅ Downloaded {len(df)} records")
                print(f"✅ Saved to {output_file}")
                return True
            else:
                print("❌ No data received from API")
                return False
        else:
            print("❌ DataFetcher not available - check Alpaca API credentials")
            return False

    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download BTC/USD historical data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', required=True,
                       choices=['5Min', '15Min', '1Hour', '4Hour'],
                       help='Timeframe to download')

    args = parser.parse_args()

    success = download_btc_data(
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe
    )

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()