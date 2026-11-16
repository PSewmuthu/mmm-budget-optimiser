"""One-command pipeline. Add each stage as its owner's PR merges."""
import runpy

STAGES = [
    ("M1 data cleaning", "src.data_prep"),
    ("M2 data analysis and transformations", "src.m2_analysis"),
    ("M3 model fit", "src.mmm_fit"),
    ("M3 model review", "src.mmm_alpha_review"),
    ("M4 benchmarking", "src.m4_m3_comparison"),
    ("M5 Channel Effectiveness and ROI tables", "src.m5_channel_effectiveness"),
    ("M6 optimisation scenarios", "src.optimise_run")
]

if __name__ == "__main__":
    for name, module in STAGES:
        print("\n")
        print("=" * 54)
        print(f"||{name.center(50, ' ')}||")
        print("=" * 54, "\n")

        runpy.run_module(module, run_name="__main__")
