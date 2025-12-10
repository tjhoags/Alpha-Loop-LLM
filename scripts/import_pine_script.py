"""
Manual Pine Script Importer
Author: Tom Hogan | Alpha Loop Capital, LLC

Since TradingView blocks scraping, this allows manual import:
1. Copy Pine Script code from TradingView
2. Save to data/tradingview/pine_scripts/ folder
3. Run this script to parse and convert to Python
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.tradingview_pine_parser import PineScriptParser


def import_pine_script_file(filepath: Path, parser: PineScriptParser):
    """Import a single Pine Script file"""
    print(f"\nProcessing: {filepath.name}")
    print("-" * 60)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            pine_code = f.read()

        # Parse Pine Script
        strategy = parser.parse(pine_code)

        print(f"Title: {strategy.title}")
        print(f"Version: {strategy.version}")
        print(f"Inputs: {len(strategy.inputs)}")
        print(f"Indicators: {len(strategy.indicators)}")
        print(f"Entry Conditions: {len(strategy.entry_conditions)}")
        print(f"Exit Conditions: {len(strategy.exit_conditions)}")

        # Convert to Python
        python_code = parser.to_python_strategy(strategy)

        # Save Python version
        output_dir = Path("src/strategies/tradingview_imported")
        output_dir.mkdir(parents=True, exist_ok=True)

        python_filename = filepath.stem + "_converted.py"
        output_path = output_dir / python_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)

        print(f"[OK] Python strategy saved to: {output_path}")

        return {
            'source_file': str(filepath),
            'output_file': str(output_path),
            'title': strategy.title,
            'version': strategy.version,
            'indicators': [{'name': i.name, 'params': i.parameters} for i in strategy.indicators],
            'entry_conditions': len(strategy.entry_conditions),
            'exit_conditions': len(strategy.exit_conditions),
            'parsed_at': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"[ERROR] Failed to parse: {e}")
        return None


def main():
    print("=" * 80)
    print("PINE SCRIPT MANUAL IMPORTER")
    print("=" * 80)
    print()
    print("Place Pine Script files (.txt or .pine) in:")
    print("  data/tradingview/pine_scripts/")
    print()

    parser = PineScriptParser()

    # Find all Pine Script files
    pine_dir = Path("data/tradingview/pine_scripts")
    pine_dir.mkdir(parents=True, exist_ok=True)

    pine_files = list(pine_dir.glob("*.txt")) + list(pine_dir.glob("*.pine"))

    if not pine_files:
        print("[INFO] No Pine Script files found.")
        print()
        print("Instructions:")
        print("1. Log into your TradingView account")
        print("2. Open a script/indicator you want to import")
        print("3. Copy the Pine Script code")
        print("4. Save to: data/tradingview/pine_scripts/script_name.txt")
        print("5. Run this script again")
        print()
        print("Example scripts to grab:")
        print("  - Volume Profile")
        print("  - Order Flow / Delta Volume")
        print("  - Market Profile")
        print("  - Anchored VWAP")
        print("  - Smart Money Concepts")
        print("  - Liquidity Sweeps")
        print("  - Order Blocks")
        print("  - Fair Value Gaps")
        return

    print(f"Found {len(pine_files)} Pine Script files\n")

    results = []
    for filepath in pine_files:
        result = import_pine_script_file(filepath, parser)
        if result:
            results.append(result)

    # Save manifest
    if results:
        manifest_file = Path("data/tradingview/import_manifest.json")
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump({
                'imported_at': datetime.now().isoformat(),
                'total_scripts': len(results),
                'scripts': results
            }, f, indent=2)

        print("\n" + "=" * 80)
        print(f"IMPORT COMPLETE: {len(results)}/{len(pine_files)} scripts")
        print("=" * 80)
        print(f"\nPython strategies saved to: src/strategies/tradingview_imported/")
        print(f"Manifest saved to: {manifest_file}")


if __name__ == "__main__":
    main()
