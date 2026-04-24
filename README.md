# Light Circle Editor

Interactive tool for designing green-light intensity patterns on two circular rings (left and right). Each ring contains **N equally spaced slots**. Clicking a slot sets its green intensity (0–255).

The resulting distributions are visualized as **von Mises bands** around each ring, with **three summary arrows** displayed per circle.

## Arrows Explained

- **White Arrow**  
  Points to the brightest slot.  
  Length represents how strongly it stands out relative to the mean.

- **Amber Arrow**  
  Represents the intensity-weighted vector sum (center of mass of the light distribution).

- **Pink Arrow**  
  Indicates the optimal rotation required to align the left distribution with the right.  
  - Computed using circular cross-correlation (FFT)  
  - Length encodes alignment quality (Cauchy–Schwarz normalized score)  
  - Multiple arrows appear if the alignment is perfectly ambiguous  

---

## Usage

```bash
python light_circle_editor.py
```

---

## Requirements

- Python 3.9+
- numpy  
  ```bash
  pip install numpy
  ```
- tkinter

### Linux users

If tkinter is missing:

```bash
sudo apt install python3-tk
```

---

## Import / Export

Patterns are stored as JSON files containing:

- `n_slots` – number of slots per ring  
- `kappa` – concentration parameter for the von Mises distribution  
- Per-slot intensity values  

Use the **IMPORT** and **EXPORT** buttons in the toolbar to load and save patterns.
