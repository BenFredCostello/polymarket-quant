def run_simplified_backtest(limit: int = 300) -> pd.DataFrame:
    """Simplified backtest that only tests model calibration, not trading."""
    
    # Fetch resolved contracts
    resolved = fetch_resolved_markets(limit=limit)
    
    records = []
    
    for market in resolved:
        if not market.strike:
            continue
            
        # Get historical data at expiry - 7 days
        expiry = _resolve_expiry(market)
        if not expiry:
            continue
            
        pricing_date = expiry - timedelta(days=7)
        inputs = get_historical_inputs(lookup, market.asset, pricing_date)
        if not inputs:
            continue
            
        S, sigma = inputs
        
        # Price with model
        T = 7 / 365.0
        K = market.strike
        is_put = market.option_type == "put"
        
        bs_call = bs_pricer.price(S, K, T, sigma)
        mc_out = mc_pricer.price(S, K, T, sigma, jump_params.get(market.asset))
        model_prob = (1 - bs_call) if is_put else bs_call
        model_prob = (model_prob + mc_out["price"]) / 2
        
        # Simple benchmark: 50/50 or ATM price
        benchmark_prob = 0.5
        
        outcome = 1 if market.outcome == "YES" else 0
        
        records.append({
            "asset": market.asset,
            "strike": K,
            "model_prob": model_prob,
            "benchmark_prob": benchmark_prob,
            "outcome": outcome,
            "model_correct": (model_prob > 0.5) == (outcome == 1),
            "benchmark_correct": (benchmark_prob > 0.5) == (outcome == 1),
        })
    
    df = pd.DataFrame(records)
    
    # Calculate performance
    model_accuracy = df["model_correct"].mean()
    benchmark_accuracy = df["benchmark_correct"].mean()
    brier_model = ((df["model_prob"] - df["outcome"]) ** 2).mean()
    
    print(f"\nModel Accuracy: {model_accuracy:.1%}")
    print(f"Benchmark (50/50) Accuracy: {benchmark_accuracy:.1%}")
    print(f"Model Brier Score: {brier_model:.4f}")
    
    # Plot calibration
    plot_calibration(df)
    
    return df