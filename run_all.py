"""One-command pipeline. Add each stage as its owner's PR merges."""
import runpy

STAGES = [
    # ("M1 data cleaning", "src.data_prep"),
    # ("M3 model fit", "src.mmm_fit"),
    # ("M5 ROI tables", "src.roi"),
    # ("M6 optimisation scenarios", "src.optimise_run"),
]

if __name__ == "__main__":
    for name, module in STAGES:
        print(f"== {name} ==")
        runpy.run_module(module, run_name="__main__")
