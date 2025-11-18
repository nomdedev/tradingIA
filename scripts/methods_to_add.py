    def load_selected_strategy(self):
        """Load the selected strategy"""
        strategy_name = self.strategy_combo.currentText()
        ticker = self.ticker_combo.currentText().split(" - ")[0]
        
        self.current_ticker = ticker
        
        strategies_info = {
            "RSI Mean Reversion": {
                "description": "Compra cuando RSI < 30 (sobrevendido), vende cuando RSI > 70 (sobrecomprado).",
                "parameters": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30, "take_profit": 2.0, "stop_loss": 1.5}
            },
            "MACD Momentum": {
                "description": "Compra en cruce alcista de MACD, vende en cruce bajista.",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "take_profit": 2.5, "stop_loss": 1.5}
            },
            "Bollinger Bands Breakout": {
                "description": "Compra cuando precio rompe banda superior, vende en banda inferior.",
                "parameters": {"period": 20, "std_dev": 2.0, "take_profit": 3.0, "stop_loss": 1.5}
            },
            "MA Crossover": {
                "description": "Compra cuando MA rápida cruza arriba de MA lenta.",
                "parameters": {"fast_ma": 50, "slow_ma": 200, "take_profit": 2.0, "stop_loss": 1.5}
            },
            "Volume Breakout": {
                "description": "Compra cuando volumen supera promedio.",
                "parameters": {"volume_threshold": 1.5, "breakout_period": 20, "take_profit": 3.0, "stop_loss": 2.0}
            }
        }
        
        if strategy_name in strategies_info:
            self.current_strategy["name"] = strategy_name
            self.current_strategy["description"] = strategies_info[strategy_name]["description"]
            self.current_strategy["parameters"] = strategies_info[strategy_name]["parameters"]
            
            params_text = "\n".join([f"{k}: {v}" for k, v in strategies_info[strategy_name]["parameters"].items()])
            self.strategy_info_label.setText(f" {strategy_name}\n\n{strategies_info[strategy_name]['description']}\n\n Parámetros:\n{params_text}")
        
        print(f"Loaded strategy: {strategy_name}")
        self.status_update.emit("Strategy loaded", "success")

    def on_ticker_changed(self, ticker_full):
        """Handle ticker change"""
        ticker = ticker_full.split(" - ")[0]
        self.current_ticker = ticker
        print(f"Ticker: {ticker}")

    def start_trading(self):
        """Start trading"""
        self.status_indicator.setText(" EN VIVO")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.decision_log.clear()
        self.decision_log.append(" Starting trading...\n")
        self.status_update.emit("Trading started", "success")

    def stop_trading(self):
        """Stop trading"""
        self.status_indicator.setText(" STOPPED")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.decision_log.append("\n Stopped.\n")
        self.status_update.emit("Trading stopped", "warning")

    def on_decision_made(self, decision):
        """Handle decision"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.decision_log.append(f"[{timestamp}] {decision['action']} - {decision['reason']}\n")

    def on_position_update(self, positions):
        """Handle positions"""
        self.trades_card.update_value(str(len(positions)))

    def on_connection_status(self, connected):
        """Handle connection"""
        if connected:
            self.status_indicator.setText(" CONNECTED")
        else:
            self.status_indicator.setText(" DISCONNECTED")

    def show_help(self):
        """Show help"""
        dialog = HelpDialog(self)
        dialog.exec()
