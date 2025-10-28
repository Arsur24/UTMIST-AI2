import json
from pathlib import Path
from typing import Any, Dict

# Optional GUI
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None

CONFIG_PATH = Path(__file__).with_suffix('.json')

DEFAULT_CFG: Dict[str, Any] = {
    "weights": {
        "danger_zone_reward": -20.0,
        "damage_interaction_reward": 1.0,
        "penalize_attack_reward": -0.02,
        "holding_more_than_3_keys": 0.0,
        "survival_reward": 0.1
    },
    "danger_zone": {
        "zone_penalty": 1,
        "zone_height": 3.5
    },
    "signals": {
        "on_win_reward": 50,
        "on_knockout_reward": 10,
        "on_combo_reward": 8,
        "on_equip_reward": 15,
        "on_drop_reward": -5
    },
    # When True, importing the training module will attempt to open the Tkinter GUI automatically
    "auto_launch_gui": True,
}


def _merge(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in default.items():
        if k in custom:
            if isinstance(v, dict) and isinstance(custom.get(k), dict):
                out[k] = _merge(v, custom.get(k, {}))
            else:
                out[k] = custom[k]
        else:
            out[k] = v
    return out


def _load_or_create() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return _merge(DEFAULT_CFG, cfg)
        except Exception:
            pass
    # write default
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CFG, f, indent=2)
    return dict(DEFAULT_CFG)


CFG = _load_or_create()


def save_cfg(path: str = None) -> None:
    p = Path(path) if path else CONFIG_PATH
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(CFG, f, indent=2)


def launch_slider_ui() -> None:
    """Open a simple Tkinter UI to tweak numeric values and save them to the JSON file.

    If tkinter isn't available this becomes a no-op.
    """
    if tk is None:
        print("Tkinter not available; cannot open config editor.")
        return

    root = tk.Tk()
    root.title("Reward Config Editor")

    entries = {}
    row = 0

    def add_entry(section: str, key: str):
        nonlocal row
        tk.Label(root, text=f"{section} / {key}").grid(column=0, row=row, sticky='w', padx=4, pady=2)
        orig = CFG[section][key]
        var = tk.StringVar(value=str(orig))
        entry = tk.Entry(root, textvariable=var, width=15)
        entry.grid(column=1, row=row, sticky='we', padx=4)
        entries[(section, key)] = (var, orig)
        row += 1

    # weights
    for k, v in CFG.get('weights', {}).items():
        add_entry('weights', k)

    # danger zone
    add_entry('danger_zone', 'zone_penalty')
    add_entry('danger_zone', 'zone_height')

    # signals
    for k, v in CFG.get('signals', {}).items():
        add_entry('signals', k)

    def on_save():
        for (section, key), (var, orig) in entries.items():
            val_str = var.get()
            try:
                val = float(val_str)
                # convert to int if original was int and value is integer
                if isinstance(orig, int) and float(val).is_integer():
                    CFG[section][key] = int(val)
                else:
                    CFG[section][key] = float(val)
            except ValueError:
                print(f"Warning: invalid value '{val_str}' for {section}/{key}, keeping original: {orig}")
                CFG[section][key] = orig
        save_cfg()
        root.destroy()
        print(f"Saved config to {CONFIG_PATH}")

    btn = tk.Button(root, text="Save and Close", command=on_save)
    btn.grid(column=0, row=row, columnspan=2, pady=8)

    root.columnconfigure(1, weight=1)
    root.mainloop()


if __name__ == '__main__':
    """Run this file directly to open the reward config GUI editor."""
    launch_slider_ui()

