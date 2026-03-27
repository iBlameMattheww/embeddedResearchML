# Embedded Machine Learning for Dynamical Systems

TinyML research project for **Simple Harmonic Oscillator (SHO)** dynamics on Raspberry Pi Pico, with training/export tooling in Python and embedded inference in C.

## Repo map (quick)
- `TinyML/` → Embedded firmware (Pico targets, inference loops, tests).
  - `TinyML/Target/src/` main programs + inference implementations.
  - `TinyML/Target/include/` model headers + exported quantized weights.
  - `TinyML/tests/` unit/integration tests.
- `VanillaNet/` → Vanilla model training + param/weight export pipeline.
- `Strupnet/` → Symplectic model training + param/weight export pipeline.
- `PINN/` → Physics-informed model training + param/weight export pipeline.
- `SimpleHarmonicOscillator/` → SHO simulation, generation, and visualization scripts.
- `Benchmarks/` → accuracy, timing, IID/OOD trajectory collection + verification scripts.
- `notes/` → project notes, build/flash docs, and reference PDFs.
- `CMakeLists.txt` → firmware build targets: `pico_vnn`, `pico_snn`, `pico_pinn`.
- `requirements.txt` → Python dependencies for training/data/benchmark scripts.
