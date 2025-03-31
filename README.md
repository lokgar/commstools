# CommsTools

The toolkit enables systematic research flow and Design of Experiments (DoE) in digital communications. Provides tools like sequence preparation, waveform generation, channel simulation, DSP for signal recovery, and more, driven by configuration (YAML/Pydantic) and modular workflows to ensure reproducibility and traceability.

DSP blocks and other computation-intensive tasks are implemented using JAX for performance and GPU acceleration.

Scope: Computational tasks only (no hardware control).

## Recommended Project Structure

For small simulations and other routines the toolkit can be used within python scripts or notebooks.

However, to fully realize potential of the toolkit for larger projects, especially those involving multiple experiments or configurations, we recommend a structured approach to organizing your project files. This will help maintain clarity and facilitate collaboration.

The following structure is suggested:

```bash
PROJECT_ROOT
├── experiments                              # Top-level directory for experiments
│  └── exp_01                                # Specific experiment
│     ├── analysis                           # Analysis notebooks/scripts and output files
│     │  ├── 00-process-captures.ipynb
│     │  └── 01-plot-constellations.ipynb
│     ├── configs                            # CommsTools setup configuration files
│     │  ├── rx.yaml
│     │  └── tx.yaml
│     ├── data                               # All data files
│     │  ├── postproc                        # Post-processed data
│     │  │  ├── recovered_symbols.npy
│     │  │  └── recovered_symbols_meta.yaml
│     │  ├── raw                             # Raw data
│     │  │  ├── capture_from_adc.bin
│     │  │  └── capture_from_adc_meta.yaml
│     │  └── waveform                        # Waveform used in the experiment 
│     │     ├── waveform_binary.bin
│     │     └── waveform_meta.yaml
│     ├── scripts                            # Scripts for running the experiment
│     │  ├── generate_waveform.py
│     │  └── process_capture.py
│     └── README.md                          # Experiment description
└── README.md                                # Top-level project description
```
