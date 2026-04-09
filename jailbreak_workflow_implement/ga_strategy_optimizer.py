"""
GA Strategy Optimizer for Multi-Stage Attack Workflow

This module implements a genetic algorithm with binary encoding to optimize
strategy selection across the multi-stage STAGE_STRATEGY_SPACE.

Chromosome encoding: 4 stages × 16 dimensions = 64 bits
Each dimension is a binary gene (0=excluded, 1=included)
"""

from typing import List, Dict, Tuple, Any, Optional
import random
import copy

# Import configuration from centralized config module
from jailbreak_workflow_implement.attack_workflow_config import (
    ALL_DIMENSIONS,
)


class GAChromosome:
    """Represents a binary-encoded strategy chromosome for multi-stage optimization."""

    def __init__(self, genes: List[int], active_stages: List[str]):
        """
        Args:
            genes: List of N bits (16 dims × num_active_stages)
                   Structure: [stage0_dim0, ..., stage0_dim15,
                               stage1_dim0, ..., stage1_dim15,
                               ...] (only for active stages)
            active_stages: List of stage names to optimize (e.g., ["context_establishment", "relationship_building"])
        """
        self.genes = genes  # List[int] where each is 0 or 1
        self.active_stages = active_stages
        self.fitness: Optional[float] = None
        self.attack_history: Optional[List[Dict[str, Any]]] = None
        self.final_result: Optional[Dict[str, Any]] = None

    def get_stage_genes(self, stage_idx: int) -> List[int]:
        """Get genes for a specific stage (0 to len(active_stages)-1)."""
        start = stage_idx * 16
        end = start + 16
        return self.genes[start:end]

    def get_selected_dimensions(self, stage_idx: int) -> List[str]:
        """Get selected dimension names for a stage."""
        stage_genes = self.get_stage_genes(stage_idx)
        return [ALL_DIMENSIONS[i] for i, gene in enumerate(stage_genes) if gene == 1]

    def get_active_stage_name(self, stage_idx: int) -> str:
        """Get the actual stage name for a given index."""
        if 0 <= stage_idx < len(self.active_stages):
            return self.active_stages[stage_idx]
        return self.active_stages[-1]

    def __repr__(self) -> str:
        num_stages = len(self.active_stages)
        return f"GAChromosome(fitness={self.fitness}, stages={num_stages}, genes={self.genes[:8]}...)"

    def copy(self) -> 'GAChromosome':
        """Create a deep copy of this chromosome."""
        new_chromo = GAChromosome(self.genes.copy(), self.active_stages)
        new_chromo.fitness = self.fitness
        new_chromo.attack_history = copy.deepcopy(self.attack_history)
        new_chromo.final_result = copy.deepcopy(self.final_result) if self.final_result else None
        return new_chromo

    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary for serialization."""
        stage_selections = {}
        for i in range(len(self.active_stages)):
            stage_name = self.get_active_stage_name(i)
            selected_dims = self.get_selected_dimensions(i)
            stage_selections[stage_name] = selected_dims

        return {
            "genes": self.genes,
            "active_stages": self.active_stages,
            "stage_selections": stage_selections,
            "fitness": self.fitness,
        }


def repair_stage_genes(genes: List[int], stage_name: str, stage_space: Dict) -> List[int]:
    """
    Repair genes to satisfy stage constraints:
    - Minimum 3 core dimensions must be selected
    - Leakage dimensions must be excluded (set to 0)

    Args:
        genes: 16 genes for one stage
        stage_name: Name of the stage (e.g., "context_establishment")
        stage_space: Strategy space dict for this stage with "core", "leakage", "optional"

    Returns:
        Repaired gene list
    """
    repaired = genes.copy()

    # Get dimension indices for this stage
    core_dims = stage_space.get("core", [])
    leakage_dims = stage_space.get("leakage", [])

    # Remove leakage dimensions
    for dim in leakage_dims:
        if dim in ALL_DIMENSIONS:
            idx = ALL_DIMENSIONS.index(dim)
            repaired[idx] = 0

    # Ensure minimum 3 core dimensions
    core_selected = sum(1 for dim in core_dims if dim in ALL_DIMENSIONS and repaired[ALL_DIMENSIONS.index(dim)] == 1)
    min_core = min(3, len(core_dims))

    if core_selected < min_core:
        # Add missing core dimensions
        for dim in core_dims:
            if dim in ALL_DIMENSIONS:
                idx = ALL_DIMENSIONS.index(dim)
                if repaired[idx] == 0:
                    repaired[idx] = 1
                    core_selected += 1
                    if core_selected >= min_core:
                        break

    return repaired


def repair_chromosome(chromosome: GAChromosome, stage_space: Dict) -> GAChromosome:
    """
    Repair all stages in a chromosome to satisfy constraints.

    Args:
        chromosome: The chromosome to repair
        stage_space: Full STAGE_STRATEGY_SPACE dict

    Returns:
        Repaired chromosome (new instance)
    """
    repaired_genes = []

    for stage_idx, stage_name in enumerate(chromosome.active_stages):
        stage_genes = chromosome.get_stage_genes(stage_idx)
        space = stage_space.get(stage_name, {})
        repaired_stage = repair_stage_genes(stage_genes, stage_name, space)
        repaired_genes.extend(repaired_stage)

    return GAChromosome(repaired_genes, chromosome.active_stages)


def crossover(parent1: GAChromosome, parent2: GAChromosome, crossover_rate: float) -> Tuple[GAChromosome, GAChromosome]:
    """
    Single-point crossover per stage.

    Args:
        parent1, parent2: Parent chromosomes
        crossover_rate: Probability of crossover (0-1)

    Returns:
        Two child chromosomes
    """
    if random.random() >= crossover_rate:
        return parent1.copy(), parent2.copy()

    # Both parents should have the same active_stages
    active_stages = parent1.active_stages
    num_stages = len(active_stages)

    child1_genes = []
    child2_genes = []

    # Crossover per stage (16 genes per stage)
    for stage in range(num_stages):
        start = stage * 16
        end = start + 16

        # Random crossover point within stage
        point = random.randint(start + 1, end - 1)

        # Perform crossover
        child1_genes.extend(parent1.genes[start:point] + parent2.genes[point:end])
        child2_genes.extend(parent2.genes[start:point] + parent1.genes[point:end])

    return GAChromosome(child1_genes, active_stages), GAChromosome(child2_genes, active_stages)


def mutate(chromosome: GAChromosome, mutation_rate: float, stage_space: Dict) -> GAChromosome:
    """
    Bit-flip mutation with constraint-aware rates:
    - Lower mutation rate for core dimensions
    - Higher mutation rate for optional dimensions

    Args:
        chromosome: The chromosome to mutate
        mutation_rate: Base mutation rate (0-1)
        stage_space: Full STAGE_STRATEGY_SPACE dict

    Returns:
        Mutated chromosome (new instance)
    """
    mutated = chromosome.genes.copy()

    for stage, stage_name in enumerate(chromosome.active_stages):
        space = stage_space.get(stage_name, {})
        core_dims = set(space.get("core", []))

        for i in range(16):
            # Adjust mutation rate based on dimension type
            dim_name = ALL_DIMENSIONS[i]
            if dim_name in core_dims:
                rate = mutation_rate * 0.3  # Lower rate for core
            else:
                rate = mutation_rate  # Full rate for optional

            if random.random() < rate:
                mutated[stage * 16 + i] = 1 - mutated[stage * 16 + i]  # Flip bit

    return GAChromosome(mutated, chromosome.active_stages)


def select(population: List[GAChromosome], top_k: int = 5) -> GAChromosome:
    """
    Fitness-proportional selection from top-k individuals.

    Args:
        population: List of chromosomes with fitness values
        top_k: Number of top individuals to select from

    Returns:
        Selected chromosome
    """
    # Filter out chromosomes without fitness
    valid_pop = [p for p in population if p.fitness is not None]

    if not valid_pop:
        return random.choice(population)

    # Sort by fitness (descending)
    sorted_pop = sorted(valid_pop, key=lambda x: x.fitness, reverse=True)

    # Select from top-k
    top_pop = sorted_pop[:top_k]
    total_fitness = sum(p.fitness for p in top_pop)

    if total_fitness == 0:
        return random.choice(top_pop)

    pick = random.uniform(0, total_fitness)
    current = 0

    for individual in top_pop:
        current += individual.fitness
        if current > pick:
            return individual

    return top_pop[0]


def initialize_population(population_size: int, stage_space: Dict, active_stages: Optional[List[str]] = None) -> List[GAChromosome]:
    """
    Initialize population with random valid chromosomes.

    Args:
        population_size: Number of individuals to create
        stage_space: Full STAGE_STRATEGY_SPACE dict
        active_stages: List of stage names to optimize (if None, uses all stages)

    Returns:
        List of initialized chromosomes
    """
    active_stages = active_stages
    population = []
    seen = set()

    while len(population) < population_size:
        genes = []

        for stage_name in active_stages:
            space = stage_space.get(stage_name, {})

            # Random initialization
            stage_genes = [random.choice([0, 1]) for _ in range(16)]

            # Repair to satisfy constraints
            stage_genes = repair_stage_genes(stage_genes, stage_name, space)

            genes.extend(stage_genes)

        # Avoid duplicates
        genes_tuple = tuple(genes)
        if genes_tuple not in seen:
            seen.add(genes_tuple)
            population.append(GAChromosome(list(genes_tuple), active_stages))

    return population


def get_best_individual(population: List[GAChromosome]) -> Optional[GAChromosome]:
    """
    Get the individual with the highest fitness from the population.

    Args:
        population: List of chromosomes

    Returns:
        Best chromosome or None if no fitness values
    """
    valid_pop = [p for p in population if p.fitness is not None]
    if not valid_pop:
        return None
    return max(valid_pop, key=lambda x: x.fitness)


def print_chromosome(chromosome: Optional[GAChromosome], label: str = "") -> None:
    """
    Pretty print a chromosome's strategy selections.

    Args:
        chromosome: The chromosome to print (can be None)
        label: Optional label to display
    """
    if chromosome is None:
        if label:
            print(f"\n{label}")
        print("  Chromosome: None")
        return

    if label:
        print(f"\n{label}")
    print(f"  Fitness: {chromosome.fitness}")
    for stage_idx, stage_name in enumerate(chromosome.active_stages):
        selected = chromosome.get_selected_dimensions(stage_idx)
        print(f"  {stage_name}: {selected}")
