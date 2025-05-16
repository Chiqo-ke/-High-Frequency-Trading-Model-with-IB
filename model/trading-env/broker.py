from flask import Flask, request, jsonify
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List

app = Flask(__name__)

@dataclass
class Position:
    id: str
    symbol: str
    lot_size: float
    is_buy: bool
    entry_price: float
    stop_loss: float
    open_time: datetime
    margin: float

class TradingAccount:
    def __init__(self, initial_balance: float = 10000.0, leverage: float = 100.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0
        self.leverage = leverage
        self.positions: List[Position] = []
        self.spread = 0.00009  # 0.9 pips for EUR/USD
        self.pip_value = 1.0  # $1 per pip for 0.1 lot
        self.lot_size = 0.1  # Mini lot: 10,000 units

    def open_position(self, symbol: str, is_buy: bool, entry_price: float):
        # Calculate margin requirement
        position_value = self.lot_size * 100000  # Value for 0.1 lot
        margin_required = position_value / self.leverage
        if margin_required > self.balance:
            return {"error": "Insufficient margin"}, 400

        # Adjust entry price for spread
        adjusted_entry = entry_price if is_buy else entry_price - self.spread
        # Set stop-loss: 15 pips below (buy) or above (sell)
        stop_loss = adjusted_entry - 0.0015 if is_buy else adjusted_entry + 0.0015

        # Create position
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            lot_size=self.lot_size,
            is_buy=is_buy,
            entry_price=adjusted_entry,
            stop_loss=stop_loss,
            open_time=datetime.now(),
            margin=margin_required
        )
        self.positions.append(position)
        self.margin_used += margin_required
        self.balance -= margin_required  # Lock margin
        return {"position_id": position.id, "entry_price": adjusted_entry, "stop_loss": stop_loss}, 200

    def close_position(self, position_id: str, close_price: float):
        position = next((p for p in self.positions if p.id == position_id), None)
        if not position:
            return {"error": "Position not found"}, 404

        # Adjust close price for spread
        adjusted_close = close_price - self.spread if position.is_buy else close_price

        # Check if stop-loss is hit
        if (position.is_buy and close_price <= position.stop_loss) or \
           (not position.is_buy and close_price >= position.stop_loss):
            adjusted_close = position.stop_loss

        # Calculate P&L
        price_diff = (adjusted_close - position.entry_price) if position.is_buy else (position.entry_price - adjusted_close)
        pips = price_diff / 0.0001  # Convert to pips
        profit = pips * self.pip_value * position.lot_size

        # Update account
        self.balance += profit + position.margin  # Release margin and add P&L
        self.margin_used -= position.margin
        self.positions.remove(position)

        # Update equity
        self.equity = self.balance + sum(
            ((p.entry_price - (self.spread if p.is_buy else 0)) - p.entry_price if p.is_buy else 
             p.entry_price - (p.entry_price + self.spread)) * 10000 * p.lot_size 
            for p in self.positions
        )

        return {
            "position_id": position_id,
            "profit_loss": profit,
            "balance": self.balance,
            "equity": self.equity
        }, 200

# Initialize account
account = TradingAccount()

@app.route('/open_trade', methods=['POST'])
def open_trade():
    data = request.get_json()
    symbol = data.get('symbol', 'EURUSD')
    action = data.get('action')  # 'buy' or 'sell'
    entry_price = data.get('entry_price', float)

    if symbol != 'EURUSD':
        return jsonify({"error": "Only EURUSD is supported"}), 400
    if action not in ['buy', 'sell']:
        return jsonify({"error": "Invalid action. Use 'buy' or 'sell'"}), 400
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        return jsonify({"error": "Invalid entry price"}), 400

    is_buy = action == 'buy'
    result, status = account.open_position(symbol, is_buy, entry_price)
    return jsonify(result), status

@app.route('/close_trade', methods=['POST'])
def close_trade():
    data = request.get_json()
    position_id = data.get('position_id')
    close_price = data.get('close_price', float)

    if not position_id:
        return jsonify({"error": "Position ID required"}), 400
    if not isinstance(close_price, (int, float)) or close_price <= 0:
        return jsonify({"error": "Invalid close price"}), 400

    result, status = account.close_position(position_id, close_price)
    return jsonify(result), status

if __name__ == '__main__':
    app.run(debug=True, port=5000)