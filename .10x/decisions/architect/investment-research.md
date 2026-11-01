# Investment research

Use sibling asset_pricing modules: backtest, allocation, technical, fundamentals, beliefs. Dependency direction: technical -> backtest; backtest -> allocation only through user policies; all -> shared result/report and validation helpers. ADR 001 defines timing and cost accounting.
