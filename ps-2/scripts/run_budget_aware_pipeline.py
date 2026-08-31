#!/usr/bin/env python3
"""
Phase 10: Compute Budgeting and Resource-Aware Resolution Allocation Script.

Executes adaptive variable-resolution mapping with compute and memory budget constraints.
Demonstrates priority-preserving resolution downgrading across multiple resource scenarios.

Usage:
  python scripts/run_budget_aware_pipeline.py
  python scripts/run_budget_aware_pipeline.py --scenario moderate
  python scripts/run_budget_aware_pipeline.py --scenario tight
  python scripts/run_budget_aware_pipeline.py --scenario impossible
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.budget.config import BudgetConfig
from backend.budget.optimizer import BudgetAwareResolutionOptimizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Budget-Aware Adaptive Resolution Optimization Pipeline."
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="all",
        choices=["all", "generous", "moderate", "tight", "impossible"],
        help="Budget scenario to evaluate ('all', 'generous', 'moderate', 'tight', 'impossible')",
    )
    return parser.parse_args()


def run_scenario(name: str, config: BudgetConfig, raw_frame, preprocessor, builder, terrain_analyzer, selector, adaptive_builder):
    print("----------------------------------------------------------------------------------")
    print(f"SCENARIO: {name.upper()}")
    print("----------------------------------------------------------------------------------")
    print("Configured Resource Limits:")
    print(f"  Max Memory           : {config.max_memory_mb:.2f} MB" if config.max_memory_mb else "  Max Memory           : None")
    print(f"  Max Allocated Cells  : {config.max_allocated_cells:,} cells" if config.max_allocated_cells else "  Max Allocated Cells  : None")
    print(f"  Max Compute Cost     : {config.max_compute_cost:,.1f} units" if config.max_compute_cost else "  Max Compute Cost     : None")

    # 1. Pipeline Execution
    prep_frame = preprocessor.process(raw_frame)
    base_grid = builder.build_map(prep_frame)
    terrain_result = terrain_analyzer.analyze(base_grid)
    decision = selector.select_resolution(base_grid, terrain_result)

    # 2. Budget Optimization
    optimizer = BudgetAwareResolutionOptimizer(config=config)
    budget_result = optimizer.optimize(
        decision=decision,
        terrain_result=terrain_result,
    )

    orig_est = budget_result.original_estimate
    opt_est = budget_result.optimized_estimate
    orig_counts = budget_result.original_counts
    opt_counts = budget_result.optimized_counts

    print("\n--- Allocation Comparison ---")
    print(f"{'Tier':<14} {'Original Cells':>16} {'Optimized Cells':>16} {'Delta':>10}")
    print("-" * 58)
    for tier in ["ultra_fine", "fine", "medium", "coarse"]:
        oc = orig_counts.get(tier, 0)
        nc = opt_counts.get(tier, 0)
        delta = nc - oc
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"{tier:<14} {oc:>16,} {nc:>16,} {delta_str:>10}")

    print("\n--- Resource Metrics & Budget Compliance ---")
    print(f"  Original Fits Budget : {budget_result.original_fits_budget}")
    print(f"  Optimized Fits Budget: {budget_result.optimized_fits_budget}")
    print(f"  Optimization Status  : {budget_result.status}")
    print(f"  Downgraded Cells     : {budget_result.num_downgraded_cells:,} ({budget_result.downgrade_percentage:.1f}%)")
    if budget_result.downgrade_counts_by_tier:
        print(f"  Downgrade Breakdown  : {budget_result.downgrade_counts_by_tier}")

    print("\n--- Resource Footprint Delta ---")
    print(f"  Allocated Cells      : {orig_est.allocated_cells:,} -> {opt_est.allocated_cells:,} ({opt_est.allocated_cells - orig_est.allocated_cells:+,} cells)")
    print(f"  Memory Footprint     : {orig_est.memory_mb:.3f} MB -> {opt_est.memory_mb:.3f} MB ({budget_result.memory_reduction_percentage:.1f}% reduction)")
    print(f"  Compute Cost Units   : {orig_est.compute_cost:,.1f} -> {opt_est.compute_cost:,.1f}")

    if budget_result.reasons:
        print(f"  Diagnostics / Notes  : {', '.join(budget_result.reasons)}")

    # 3. Adaptive Map Construction with Budget-Constrained Decisions
    if budget_result.optimized_fits_budget or budget_result.status == "impossible_budget":
        adaptive_map = adaptive_builder.build(prep_frame, budget_result.optimized_decision)
        print(f"\n[*] Constructed AdaptiveMap25D with {len(adaptive_map.tier_maps)} active tiers ({adaptive_map.num_represented_cells:,} represented cells)")

    print("")


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    prep_cfg = PreprocessingConfig.from_yaml(config_path) if config_path.is_file() else PreprocessingConfig()
    map_cfg = MappingConfig.from_yaml(config_path) if config_path.is_file() else MappingConfig()
    terrain_cfg = TerrainConfig.from_yaml(config_path) if config_path.is_file() else TerrainConfig()
    priority_cfg = PriorityConfig.from_yaml(config_path) if config_path.is_file() else PriorityConfig()

    preprocessor = PreprocessingPipeline(config=prep_cfg)
    builder = Uniform25DMapBuilder(config=map_cfg)
    terrain_analyzer = TerrainAnalyzer(config=terrain_cfg)
    selector = AdaptiveResolutionSelector(config=priority_cfg)
    adaptive_builder = AdaptiveMap25DBuilder(config=map_cfg)

    generator = SyntheticLidarGenerator(seed=42)
    raw_frame = generator.generate_frame(frame_id=0, num_ground_points=6000)

    print("==================================================================================")
    print("ADAPTIVE LiDAR MAPPING — PHASE 10 COMPUTE BUDGETING & RESOLUTION ALLOCATION")
    print("==================================================================================")
    print(f"Raw Input Points    : {raw_frame.num_points:,}")
    print(f"Base Grid Resolution: {map_cfg.resolution:.2f} m/cell")
    print("==================================================================================\n")

    scenarios = {
        "generous": BudgetConfig(
            enabled=True,
            max_memory_mb=25.0,
            max_allocated_cells=500000,
            max_compute_cost=10000.0,
        ),
        "moderate": BudgetConfig(
            enabled=True,
            max_memory_mb=1.60,
            max_allocated_cells=66000,
            max_compute_cost=3000.0,
        ),
        "tight": BudgetConfig(
            enabled=True,
            max_memory_mb=1.50,
            max_allocated_cells=64000,
            max_compute_cost=1200.0,
        ),
        "impossible": BudgetConfig(
            enabled=True,
            max_memory_mb=1.00,
            max_allocated_cells=50000,
            max_compute_cost=500.0,
        ),
    }

    if args.scenario == "all":
        for s_name, s_cfg in scenarios.items():
            run_scenario(s_name, s_cfg, raw_frame, preprocessor, builder, terrain_analyzer, selector, adaptive_builder)
    else:
        run_scenario(args.scenario, scenarios[args.scenario], raw_frame, preprocessor, builder, terrain_analyzer, selector, adaptive_builder)

    print("==================================================================================")
    print("Budget-aware adaptive mapping pipeline complete.")
    print("==================================================================================")


if __name__ == "__main__":
    main()
