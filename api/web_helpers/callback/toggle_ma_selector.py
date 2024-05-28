def toggle_ma_selector(selected_data: list[str]) -> dict[str, str]:
    if "SMA" in selected_data or "EWMA" in selected_data:
        return {"display": "block"}
    else:
        return {"display": "none"}
