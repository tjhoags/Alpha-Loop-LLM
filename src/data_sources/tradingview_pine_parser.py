"""
TradingView Pine Script Parser
Author: Tom Hogan | Alpha Loop Capital, LLC

Parses Pine Script code and converts to Python trading logic.
Extracts indicators, conditions, and strategy logic.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PineIndicator:
    """Parsed Pine Script indicator"""
    name: str
    function: str
    parameters: Dict[str, Any]
    output_variables: List[str]


@dataclass
class PineCondition:
    """Trading condition from Pine Script"""
    type: str  # 'entry', 'exit', 'filter'
    direction: str  # 'long', 'short'
    logic: str
    variables_used: List[str]


@dataclass
class PineStrategy:
    """Complete parsed Pine Script strategy"""
    title: str
    version: str
    indicators: List[PineIndicator] = field(default_factory=list)
    entry_conditions: List[PineCondition] = field(default_factory=list)
    exit_conditions: List[PineCondition] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)


class PineScriptParser:
    """
    Parses Pine Script and extracts:
    - Indicator definitions (SMA, EMA, RSI, etc.)
    - Entry/exit conditions
    - Strategy parameters
    - Variable assignments
    """

    # Common Pine Script functions
    INDICATOR_PATTERNS = {
        'sma': r'sma\(([^,]+),\s*(\d+)\)',
        'ema': r'ema\(([^,]+),\s*(\d+)\)',
        'rsi': r'rsi\(([^,]+),\s*(\d+)\)',
        'macd': r'macd\(([^,]+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
        'bbands': r'bb\(([^,]+),\s*(\d+),\s*([0-9.]+)\)',
        'atr': r'atr\((\d+)\)',
        'stoch': r'stoch\(([^,]+),\s*([^,]+),\s*([^,]+),\s*(\d+),\s*(\d+)\)',
        'vwap': r'vwap\(([^)]+)\)',
        'volume': r'volume',
        'highest': r'highest\(([^,]+),\s*(\d+)\)',
        'lowest': r'lowest\(([^,]+),\s*(\d+)\)',
        'crossover': r'crossover\(([^,]+),\s*([^)]+)\)',
        'crossunder': r'crossunder\(([^,]+),\s*([^)]+)\)',
    }

    def __init__(self):
        self.current_strategy = None

    def parse(self, pine_code: str) -> PineStrategy:
        """
        Parse complete Pine Script code

        Args:
            pine_code: Pine Script source code

        Returns:
            PineStrategy object with extracted logic
        """
        strategy = PineStrategy(title="Unknown", version="5")

        # Extract title
        title_match = re.search(r'title\s*=\s*["\']([^"\']+)["\']', pine_code)
        if title_match:
            strategy.title = title_match.group(1)

        # Extract version
        version_match = re.search(r'//@version\s*=\s*(\d+)', pine_code)
        if version_match:
            strategy.version = version_match.group(1)

        # Extract inputs
        strategy.inputs = self._parse_inputs(pine_code)

        # Extract variable assignments
        strategy.variables = self._parse_variables(pine_code)

        # Extract indicators
        strategy.indicators = self._parse_indicators(pine_code)

        # Extract entry conditions
        strategy.entry_conditions = self._parse_conditions(pine_code, 'entry')

        # Extract exit conditions
        strategy.exit_conditions = self._parse_conditions(pine_code, 'exit')

        return strategy

    def _parse_inputs(self, code: str) -> Dict[str, Any]:
        """Extract input parameters"""
        inputs = {}

        # Match: input.int(14, "RSI Period")
        int_inputs = re.findall(
            r'(\w+)\s*=\s*input\.int\(([^,]+),\s*["\']([^"\']+)["\']',
            code
        )
        for var_name, default_val, label in int_inputs:
            inputs[var_name] = {
                'type': 'int',
                'default': int(default_val),
                'label': label
            }

        # Match: input.float(0.02, "Step")
        float_inputs = re.findall(
            r'(\w+)\s*=\s*input\.float\(([^,]+),\s*["\']([^"\']+)["\']',
            code
        )
        for var_name, default_val, label in float_inputs:
            inputs[var_name] = {
                'type': 'float',
                'default': float(default_val),
                'label': label
            }

        # Match: input.bool(true, "Enable Filter")
        bool_inputs = re.findall(
            r'(\w+)\s*=\s*input\.bool\((\w+),\s*["\']([^"\']+)["\']',
            code
        )
        for var_name, default_val, label in bool_inputs:
            inputs[var_name] = {
                'type': 'bool',
                'default': default_val.lower() == 'true',
                'label': label
            }

        return inputs

    def _parse_variables(self, code: str) -> Dict[str, str]:
        """Extract variable assignments"""
        variables = {}

        # Match: var_name = expression
        var_assigns = re.findall(r'(\w+)\s*=\s*([^/\n]+)', code)

        for var_name, expression in var_assigns:
            expression = expression.strip()
            if var_name not in ['strategy', 'indicator', 'plot']:
                variables[var_name] = expression

        return variables

    def _parse_indicators(self, code: str) -> List[PineIndicator]:
        """Extract indicator calculations"""
        indicators = []

        for indicator_name, pattern in self.INDICATOR_PATTERNS.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)

            for match in matches:
                # Extract variable name (if assigned)
                var_match = re.search(
                    rf'(\w+)\s*=\s*{re.escape(match.group(0))}',
                    code
                )
                var_name = var_match.group(1) if var_match else f"{indicator_name}_output"

                params = {}
                if indicator_name == 'sma' or indicator_name == 'ema':
                    params = {'source': match.group(1), 'length': int(match.group(2))}
                elif indicator_name == 'rsi':
                    params = {'source': match.group(1), 'length': int(match.group(2))}
                elif indicator_name == 'macd':
                    params = {
                        'source': match.group(1),
                        'fast': int(match.group(2)),
                        'slow': int(match.group(3)),
                        'signal': int(match.group(4))
                    }

                indicators.append(PineIndicator(
                    name=indicator_name,
                    function=match.group(0),
                    parameters=params,
                    output_variables=[var_name]
                ))

        return indicators

    def _parse_conditions(self, code: str, condition_type: str) -> List[PineCondition]:
        """
        Extract entry/exit conditions

        Args:
            code: Pine Script code
            condition_type: 'entry' or 'exit'
        """
        conditions = []

        if condition_type == 'entry':
            # Match: strategy.entry("Long", strategy.long, when=condition)
            long_entries = re.findall(
                r'strategy\.entry\(["\']([^"\']+)["\'],\s*strategy\.long(?:,\s*when\s*=\s*([^)]+))?\)',
                code
            )
            for label, condition in long_entries:
                conditions.append(PineCondition(
                    type='entry',
                    direction='long',
                    logic=condition if condition else 'true',
                    variables_used=self._extract_variables(condition or '')
                ))

            # Match: strategy.entry("Short", strategy.short, when=condition)
            short_entries = re.findall(
                r'strategy\.entry\(["\']([^"\']+)["\'],\s*strategy\.short(?:,\s*when\s*=\s*([^)]+))?\)',
                code
            )
            for label, condition in short_entries:
                conditions.append(PineCondition(
                    type='entry',
                    direction='short',
                    logic=condition if condition else 'true',
                    variables_used=self._extract_variables(condition or '')
                ))

        elif condition_type == 'exit':
            # Match: strategy.exit(...) or strategy.close(...)
            exits = re.findall(
                r'strategy\.(?:exit|close)\([^)]+when\s*=\s*([^)]+)\)',
                code
            )
            for condition in exits:
                conditions.append(PineCondition(
                    type='exit',
                    direction='both',
                    logic=condition,
                    variables_used=self._extract_variables(condition)
                ))

        return conditions

    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from an expression"""
        # Find all word characters that aren't Pine keywords
        pine_keywords = {'and', 'or', 'not', 'true', 'false', 'if', 'else', 'for', 'while'}
        variables = re.findall(r'\b([a-zA-Z_]\w*)\b', expression)
        return [v for v in variables if v not in pine_keywords]

    def to_python_strategy(self, strategy: PineStrategy) -> str:
        """
        Convert parsed Pine Strategy to Python strategy function

        Returns:
            Python code as string
        """
        python_code = f'''"""
{strategy.title}
Converted from Pine Script
"""

import pandas as pd
import numpy as np
from typing import Dict

def {self._sanitize_name(strategy.title)}(price_data: pd.DataFrame, **params) -> Dict[str, float]:
    """
    {strategy.title}

    Parameters:
'''

        # Add inputs
        for var_name, config in strategy.inputs.items():
            default = config['default']
            python_code += f"        {var_name}: {config['type']} = {default}  # {config['label']}\n"

        python_code += '    """\n\n'

        # Add indicator calculations
        python_code += "    # Calculate indicators\n"
        for indicator in strategy.indicators:
            if indicator.name == 'sma':
                python_code += f"    {indicator.output_variables[0]} = price_data['{indicator.parameters['source']}'].rolling({indicator.parameters['length']}).mean()\n"
            elif indicator.name == 'ema':
                python_code += f"    {indicator.output_variables[0]} = price_data['{indicator.parameters['source']}'].ewm(span={indicator.parameters['length']}).mean()\n"
            elif indicator.name == 'rsi':
                python_code += f"    # RSI calculation\n"
                python_code += f"    delta = price_data['{indicator.parameters['source']}'].diff()\n"
                python_code += f"    gain = (delta.where(delta > 0, 0)).rolling({indicator.parameters['length']}).mean()\n"
                python_code += f"    loss = (-delta.where(delta < 0, 0)).rolling({indicator.parameters['length']}).mean()\n"
                python_code += f"    rs = gain / loss\n"
                python_code += f"    {indicator.output_variables[0]} = 100 - (100 / (1 + rs))\n"

        python_code += "\n    # Entry conditions\n"
        python_code += "    signals = {}\n"

        # Add entry logic
        for condition in strategy.entry_conditions:
            direction = condition.direction
            logic = condition.logic.replace('and', '&').replace('or', '|')
            python_code += f"    # {direction.upper()} entry: {logic}\n"
            python_code += f"    # TODO: Implement condition logic\n\n"

        python_code += "    return signals\n"

        return python_code

    def _sanitize_name(self, name: str) -> str:
        """Convert strategy name to valid Python function name"""
        # Remove special characters, replace spaces with underscores
        name = re.sub(r'[^\w\s]', '', name)
        name = name.replace(' ', '_').lower()
        return name


def main():
    """Example usage"""

    # Example Pine Script
    pine_code = """
//@version=5
strategy("RSI + Moving Average Crossover", overlay=true)

// Inputs
rsi_length = input.int(14, "RSI Length")
sma_fast = input.int(10, "Fast SMA")
sma_slow = input.int(30, "Slow SMA")
rsi_oversold = input.int(30, "RSI Oversold")
rsi_overbought = input.int(70, "RSI Overbought")

// Indicators
rsi_value = ta.rsi(close, rsi_length)
sma_fast_value = ta.sma(close, sma_fast)
sma_slow_value = ta.sma(close, sma_slow)

// Entry Conditions
long_condition = ta.crossover(sma_fast_value, sma_slow_value) and rsi_value < rsi_overbought
short_condition = ta.crossunder(sma_fast_value, sma_slow_value) and rsi_value > rsi_oversold

// Entries
if long_condition
    strategy.entry("Long", strategy.long)

if short_condition
    strategy.entry("Short", strategy.short)

// Exit on opposite signal
if short_condition
    strategy.close("Long")

if long_condition
    strategy.close("Short")
"""

    parser = PineScriptParser()
    strategy = parser.parse(pine_code)

    print("="*60)
    print("PINE SCRIPT PARSER")
    print("="*60)
    print(f"\nStrategy: {strategy.title}")
    print(f"Version: {strategy.version}")
    print(f"\nInputs: {len(strategy.inputs)}")
    for name, config in strategy.inputs.items():
        print(f"  - {name}: {config['type']} = {config['default']} ({config['label']})")

    print(f"\nIndicators: {len(strategy.indicators)}")
    for ind in strategy.indicators:
        print(f"  - {ind.name}: {ind.parameters}")

    print(f"\nEntry Conditions: {len(strategy.entry_conditions)}")
    for cond in strategy.entry_conditions:
        print(f"  - {cond.direction}: {cond.logic}")

    print(f"\nExit Conditions: {len(strategy.exit_conditions)}")
    for cond in strategy.exit_conditions:
        print(f"  - {cond.direction}: {cond.logic}")

    # Convert to Python
    print("\n" + "="*60)
    print("PYTHON CONVERSION")
    print("="*60)
    python_code = parser.to_python_strategy(strategy)
    print(python_code)


if __name__ == "__main__":
    main()
