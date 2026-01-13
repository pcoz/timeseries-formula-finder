"""
Symbolic regression using genetic programming.

This module implements a self-contained genetic programming engine
for discovering mathematical expressions that fit data.
"""

import random
from typing import List, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize

from .symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    PrimitiveSet, FitnessResult, SymbolicFitResult, SymbolicRegressionResult,
    DiscoveryMode, MODE_TO_PRIMITIVES, DOMAIN_PROBE_ORDER,
    MACRO_DEFAULTS, MINIMAL_PRIMITIVES,
)
from .detector import check_residuals_are_noise


def calculate_fitness(
    tree: ExprNode,
    x: np.ndarray,
    y: np.ndarray,
    parsimony_coefficient: float = 0.001
) -> FitnessResult:
    """
    Calculate fitness of an expression tree.

    Fitness combines accuracy (R-squared) with a complexity penalty
    to encourage simpler solutions.

    Args:
        tree: Expression tree to evaluate
        x: Input values
        y: Target values
        parsimony_coefficient: Penalty per node (higher = simpler solutions)

    Returns:
        FitnessResult with all metrics
    """
    try:
        y_pred = tree.evaluate(x)

        # Handle NaN/Inf
        if not np.all(np.isfinite(y_pred)):
            return FitnessResult(
                mse=float('inf'),
                r_squared=-float('inf'),
                complexity=tree.size(),
                depth=tree.depth(),
                score=-float('inf')
            )

        mse = float(np.mean((y - y_pred) ** 2))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        complexity = tree.size()
        depth = tree.depth()

        # Parsimony-pressure score
        score = r_squared - parsimony_coefficient * complexity

        return FitnessResult(
            mse=mse,
            r_squared=float(r_squared),
            complexity=complexity,
            depth=depth,
            score=float(score)
        )

    except Exception:
        return FitnessResult(
            mse=float('inf'),
            r_squared=-float('inf'),
            complexity=tree.size(),
            depth=tree.depth(),
            score=-float('inf')
        )


def optimize_constants(
    tree: ExprNode,
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100
) -> Tuple[ExprNode, float]:
    """
    Optimize constant values in a tree using L-BFGS-B.

    After the tree structure is evolved, this function fine-tunes
    the numeric constants to minimize MSE.

    Args:
        tree: Expression tree with constants to optimize
        x: Input data
        y: Target data
        max_iter: Maximum optimization iterations

    Returns:
        Tuple of (optimized_tree, final_mse)
    """
    tree = tree.copy()
    constants = tree.get_constants()

    if len(constants) == 0:
        y_pred = tree.evaluate(x)
        if np.all(np.isfinite(y_pred)):
            mse = float(np.mean((y - y_pred) ** 2))
        else:
            mse = float('inf')
        return tree, mse

    def objective(c: np.ndarray) -> float:
        tree.set_constants(list(c))
        y_pred = tree.evaluate(x)
        if not np.all(np.isfinite(y_pred)):
            return 1e10
        return float(np.mean((y - y_pred) ** 2))

    try:
        result = minimize(
            objective,
            x0=np.array(constants),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
        tree.set_constants(list(result.x))
        return tree, float(result.fun)
    except Exception:
        return tree, objective(np.array(constants))


class GPEngine:
    """
    Self-contained genetic programming engine.

    Implements tree-based GP with standard genetic operators:
    - Tournament selection
    - Subtree crossover
    - Multiple mutation types
    - Elitism
    - Structural diversity preservation
    """

    def __init__(
        self,
        primitives: PrimitiveSet,
        population_size: int = 500,
        generations: int = 50,
        tournament_size: int = 7,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        max_depth: int = 6,
        min_depth: int = 2,
        elitism: int = 5,
        parsimony_coefficient: float = 0.001,
        preserve_diversity: bool = True,
        preoptimize_macros: bool = True,
    ):
        self.primitives = primitives
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.elitism = elitism
        self.parsimony_coefficient = parsimony_coefficient
        self.preserve_diversity = preserve_diversity
        self.preoptimize_macros = preoptimize_macros

    def random_tree(self, depth: int, method: str = 'grow') -> ExprNode:
        """
        Generate a random expression tree.

        Args:
            depth: Maximum depth for the tree
            method: 'grow' (variable depth) or 'full' (fixed depth)

        Returns:
            Random ExprNode tree
        """
        # Terminal node condition
        if depth <= 1 or (method == 'grow' and depth > 1 and random.random() < 0.3):
            return self._random_terminal()

        # Maybe generate a macro (as a "super-terminal")
        if (self.primitives.macro_ops and
            random.random() < self.primitives.macro_probability):
            return self._random_macro()

        # Function node - choose unary or binary
        use_unary = (
            self.primitives.unary_ops and
            (not self.primitives.binary_ops or random.random() < 0.4)
        )

        if use_unary:
            op = random.choice(self.primitives.unary_ops)
            child = self.random_tree(depth - 1, method)
            return ExprNode(
                node_type=NodeType.UNARY_OP,
                unary_op=op,
                child=child
            )
        else:
            op = random.choice(self.primitives.binary_ops)
            left = self.random_tree(depth - 1, method)
            right = self.random_tree(depth - 1, method)
            return ExprNode(
                node_type=NodeType.BINARY_OP,
                binary_op=op,
                left=left,
                right=right
            )

    def _random_terminal(self) -> ExprNode:
        """Generate a random terminal node (constant, variable, or macro)"""
        # Check if we should generate a macro instead
        if (self.primitives.macro_ops and
            random.random() < self.primitives.macro_probability):
            return self._random_macro()

        # Regular terminal
        if random.random() < self.primitives.constant_probability:
            # Constant
            if random.random() < 0.3 and self.primitives.special_constants:
                value = random.choice(self.primitives.special_constants)
            else:
                value = random.uniform(*self.primitives.constant_range)
            return ExprNode(node_type=NodeType.CONSTANT, value=value)
        else:
            # Variable
            return ExprNode(node_type=NodeType.VARIABLE)

    def _random_macro(self) -> ExprNode:
        """Generate a random macro node with perturbed default parameters"""
        macro_op = random.choice(self.primitives.macro_ops)
        defaults = MACRO_DEFAULTS[macro_op]

        # Perturb default parameters slightly for diversity
        params = []
        for default in defaults:
            if default == 0:
                # For zero defaults, use small random value
                params.append(random.uniform(-0.5, 0.5))
            else:
                # Perturb by ±50%
                params.append(default * random.uniform(0.5, 1.5))

        return ExprNode(
            node_type=NodeType.MACRO,
            macro_op=macro_op,
            macro_params=params
        )

    def get_structure_signature(self, tree: ExprNode) -> str:
        """
        Extract a structural signature from a tree.

        This identifies the "type" of expression for diversity preservation.
        Trees with the same signature are considered structurally equivalent.

        Returns:
            String signature like "macro:gaussian", "unary:sin", "binary:add", etc.
        """
        if tree.node_type == NodeType.MACRO:
            return f"macro:{tree.macro_op.value}"
        elif tree.node_type == NodeType.CONSTANT:
            return "const"
        elif tree.node_type == NodeType.VARIABLE:
            return "var"
        elif tree.node_type == NodeType.UNARY_OP:
            # Include the operator and child's top-level structure
            child_sig = self.get_structure_signature(tree.child)
            return f"unary:{tree.unary_op.value}({child_sig})"
        elif tree.node_type == NodeType.BINARY_OP:
            # For binary ops, get signatures of both children
            left_sig = self.get_structure_signature(tree.left)
            right_sig = self.get_structure_signature(tree.right)
            return f"binary:{tree.binary_op.value}({left_sig},{right_sig})"
        return "unknown"

    def get_macro_signature(self, tree: ExprNode) -> Optional[str]:
        """
        Get the macro type if tree contains a macro, else None.

        This is a simpler signature for macro-focused diversity.
        """
        if tree.node_type == NodeType.MACRO:
            return tree.macro_op.value

        # Check if any subtree contains a macro
        if tree.node_type == NodeType.UNARY_OP:
            return self.get_macro_signature(tree.child)
        elif tree.node_type == NodeType.BINARY_OP:
            left_macro = self.get_macro_signature(tree.left)
            if left_macro:
                return left_macro
            return self.get_macro_signature(tree.right)

        return None

    def _create_seeded_macro(self, macro_op: MacroOp, x: np.ndarray, y: np.ndarray) -> ExprNode:
        """
        Create a macro with pre-optimized parameters.

        This gives macros a fighting chance by optimizing their
        parameters before they compete with other individuals.
        """
        defaults = MACRO_DEFAULTS[macro_op]

        # Start with perturbed defaults
        params = []
        for default in defaults:
            if default == 0:
                params.append(random.uniform(-1.0, 1.0))
            else:
                params.append(default * random.uniform(0.7, 1.3))

        tree = ExprNode(
            node_type=NodeType.MACRO,
            macro_op=macro_op,
            macro_params=params
        )

        # Pre-optimize if enabled
        if self.preoptimize_macros:
            tree, _ = optimize_constants(tree, x, y, max_iter=50)

        return tree

    def initialize_population(self, x: np.ndarray = None, y: np.ndarray = None) -> List[ExprNode]:
        """
        Create initial population using ramped half-and-half.

        This method creates trees with varying depths and structures
        to ensure diversity in the initial population.

        If x and y are provided and macros are available, seeds the
        population with pre-optimized instances of each macro type.
        """
        population = []
        depths = list(range(self.min_depth, self.max_depth + 1))

        # Seed with pre-optimized macros if data is available
        if self.primitives.macro_ops and x is not None and y is not None:
            for macro_op in self.primitives.macro_ops:
                # Add multiple instances of each macro type
                for _ in range(2):
                    tree = self._create_seeded_macro(macro_op, x, y)
                    population.append(tree)

        # Fill remaining slots with random trees
        while len(population) < self.population_size:
            depth = depths[len(population) % len(depths)]
            method = 'full' if len(population) % 2 == 0 else 'grow'
            tree = self.random_tree(depth, method)
            population.append(tree)

        return population[:self.population_size]

    def tournament_select(
        self,
        population: List[ExprNode],
        fitness_scores: List[FitnessResult]
    ) -> ExprNode:
        """Select an individual via tournament selection"""
        indices = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i].score)
        return population[best_idx].copy()

    def _get_random_subtree_parent(
        self,
        tree: ExprNode
    ) -> Tuple[Optional[ExprNode], Optional[str], ExprNode]:
        """
        Get a random subtree and its parent info.

        Returns:
            Tuple of (parent_node, child_attr_name, subtree)
            For root, parent is None and attr is None
        """
        nodes_with_parents = [(None, None, tree)]

        def collect(parent, attr, node):
            if node.node_type == NodeType.UNARY_OP:
                nodes_with_parents.append((node, 'child', node.child))
                collect(node, 'child', node.child)
            elif node.node_type == NodeType.BINARY_OP:
                nodes_with_parents.append((node, 'left', node.left))
                nodes_with_parents.append((node, 'right', node.right))
                collect(node, 'left', node.left)
                collect(node, 'right', node.right)

        collect(None, None, tree)
        return random.choice(nodes_with_parents)

    def crossover(
        self,
        parent1: ExprNode,
        parent2: ExprNode
    ) -> Tuple[ExprNode, ExprNode]:
        """
        Subtree crossover: swap random subtrees between parents.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get random subtrees
        p1_parent, p1_attr, p1_subtree = self._get_random_subtree_parent(child1)
        p2_parent, p2_attr, p2_subtree = self._get_random_subtree_parent(child2)

        # Check depth constraints before swapping
        if p1_parent is not None and p2_subtree.depth() + self._get_depth_to_node(child1, p1_parent) <= self.max_depth:
            setattr(p1_parent, p1_attr, p2_subtree.copy())
        elif p1_parent is None:
            # Swapping root
            if p2_subtree.depth() <= self.max_depth:
                child1 = p2_subtree.copy()

        if p2_parent is not None and p1_subtree.depth() + self._get_depth_to_node(child2, p2_parent) <= self.max_depth:
            setattr(p2_parent, p2_attr, p1_subtree.copy())
        elif p2_parent is None:
            if p1_subtree.depth() <= self.max_depth:
                child2 = p1_subtree.copy()

        return child1, child2

    def _get_depth_to_node(self, root: ExprNode, target: ExprNode) -> int:
        """Get the depth from root to a target node"""
        if root is target:
            return 0
        if root.node_type == NodeType.UNARY_OP:
            if root.child is target:
                return 1
            sub = self._get_depth_to_node(root.child, target)
            return 1 + sub if sub >= 0 else -1
        elif root.node_type == NodeType.BINARY_OP:
            if root.left is target:
                return 1
            if root.right is target:
                return 1
            left_depth = self._get_depth_to_node(root.left, target)
            if left_depth >= 0:
                return 1 + left_depth
            right_depth = self._get_depth_to_node(root.right, target)
            if right_depth >= 0:
                return 1 + right_depth
        return -1

    def mutate(self, tree: ExprNode) -> ExprNode:
        """
        Apply mutation to a tree.

        Mutation types:
        1. Subtree mutation: replace subtree with random tree
        2. Point mutation: change operator or constant
        3. Hoist mutation: replace tree with random subtree
        4. Constant perturbation: add noise to constants
        """
        tree = tree.copy()
        mutation_type = random.random()

        if mutation_type < 0.4:
            return self._subtree_mutation(tree)
        elif mutation_type < 0.6:
            return self._point_mutation(tree)
        elif mutation_type < 0.8:
            return self._hoist_mutation(tree)
        else:
            return self._constant_mutation(tree)

    def _subtree_mutation(self, tree: ExprNode) -> ExprNode:
        """Replace a random subtree with a new random tree"""
        parent, attr, subtree = self._get_random_subtree_parent(tree)

        if parent is None:
            # Replace entire tree
            return self.random_tree(self.max_depth, 'grow')

        # Calculate remaining depth budget
        depth_to_parent = self._get_depth_to_node(tree, parent)
        remaining_depth = max(1, self.max_depth - depth_to_parent - 1)

        new_subtree = self.random_tree(remaining_depth, 'grow')
        setattr(parent, attr, new_subtree)
        return tree

    def _point_mutation(self, tree: ExprNode) -> ExprNode:
        """Mutate a single node (change operator or constant)"""
        nodes = tree.get_all_nodes()
        if not nodes:
            return tree

        node = random.choice(nodes)

        if node.node_type == NodeType.CONSTANT:
            # Perturb constant or replace with special constant
            if random.random() < 0.3 and self.primitives.special_constants:
                node.value = random.choice(self.primitives.special_constants)
            else:
                node.value = random.uniform(*self.primitives.constant_range)

        elif node.node_type == NodeType.UNARY_OP and self.primitives.unary_ops:
            node.unary_op = random.choice(self.primitives.unary_ops)

        elif node.node_type == NodeType.BINARY_OP and self.primitives.binary_ops:
            node.binary_op = random.choice(self.primitives.binary_ops)

        elif node.node_type == NodeType.MACRO and self.primitives.macro_ops:
            # Either change macro type or perturb parameters
            if random.random() < 0.3:
                # Change macro type
                new_macro = random.choice(self.primitives.macro_ops)
                if new_macro != node.macro_op:
                    node.macro_op = new_macro
                    node.macro_params = list(MACRO_DEFAULTS[new_macro])
            else:
                # Perturb one parameter
                idx = random.randrange(len(node.macro_params))
                node.macro_params[idx] *= random.uniform(0.5, 2.0)

        return tree

    def _hoist_mutation(self, tree: ExprNode) -> ExprNode:
        """Replace tree with one of its subtrees (simplification)"""
        nodes = tree.get_all_nodes()
        if len(nodes) <= 1:
            return tree

        # Pick a non-root subtree to become the new root
        subtree = random.choice(nodes[1:]) if len(nodes) > 1 else nodes[0]
        return subtree.copy()

    def _constant_mutation(self, tree: ExprNode) -> ExprNode:
        """Add noise to all constants in the tree"""
        constants = tree.get_constants()
        if not constants:
            return tree

        # Add Gaussian noise
        noise_scale = 0.1 * (self.primitives.constant_range[1] - self.primitives.constant_range[0])
        new_constants = [c + random.gauss(0, noise_scale) for c in constants]
        tree.set_constants(new_constants)
        return tree

    def evolve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        callback: Optional[Callable[[int, FitnessResult], None]] = None
    ) -> List[Tuple[ExprNode, FitnessResult]]:
        """
        Run the evolutionary process.

        Args:
            x: Input data
            y: Target data
            callback: Optional callback(generation, best_fitness) for progress

        Returns:
            Pareto front of best solutions
        """
        # Initialize with pre-optimized macros if data available
        population = self.initialize_population(x, y)
        pareto_archive: List[Tuple[ExprNode, FitnessResult]] = []

        # Track best individual for each macro type (structural diversity)
        macro_elite: dict = {}  # macro_name -> (tree, fitness)

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for tree in population:
                fit = calculate_fitness(tree, x, y, self.parsimony_coefficient)
                fitness_scores.append(fit)

            # Update Pareto archive
            for tree, fit in zip(population, fitness_scores):
                if fit.score > -float('inf'):
                    self._update_pareto_archive(pareto_archive, tree, fit)

            # Update macro elite (best of each macro type)
            if self.preserve_diversity and self.primitives.macro_ops:
                for tree, fit in zip(population, fitness_scores):
                    if fit.score > -float('inf'):
                        macro_sig = self.get_macro_signature(tree)
                        if macro_sig:
                            if (macro_sig not in macro_elite or
                                fit.r_squared > macro_elite[macro_sig][1].r_squared):
                                macro_elite[macro_sig] = (tree.copy(), fit)

            # Report progress
            if callback:
                valid_scores = [f for f in fitness_scores if f.score > -float('inf')]
                if valid_scores:
                    best_fit = max(valid_scores, key=lambda f: f.score)
                    callback(gen, best_fit)

            # Create next generation
            next_population = []

            # Elitism: keep best individuals
            elite_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i].score,
                reverse=True
            )[:self.elitism]

            for idx in elite_indices:
                next_population.append(population[idx].copy())

            # Structural diversity: ensure best of each macro type survives
            if self.preserve_diversity:
                for macro_name, (tree, _) in macro_elite.items():
                    # Check if this macro type is already in next_population
                    already_present = any(
                        self.get_macro_signature(t) == macro_name
                        for t in next_population
                    )
                    if not already_present and len(next_population) < self.population_size:
                        next_population.append(tree.copy())

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_prob:
                    p1 = self.tournament_select(population, fitness_scores)
                    p2 = self.tournament_select(population, fitness_scores)
                    c1, c2 = self.crossover(p1, p2)

                    # Maybe mutate offspring
                    if random.random() < self.mutation_prob:
                        c1 = self.mutate(c1)
                    if random.random() < self.mutation_prob:
                        c2 = self.mutate(c2)

                    next_population.append(c1)
                    if len(next_population) < self.population_size:
                        next_population.append(c2)
                else:
                    p = self.tournament_select(population, fitness_scores)
                    if random.random() < self.mutation_prob:
                        p = self.mutate(p)
                    next_population.append(p)

            population = next_population[:self.population_size]

        # Final constant optimization for Pareto front
        optimized_front = []
        for tree, fit in pareto_archive:
            opt_tree, _ = optimize_constants(tree, x, y)
            opt_fit = calculate_fitness(opt_tree, x, y, self.parsimony_coefficient)
            optimized_front.append((opt_tree, opt_fit))

        # Also include best of each macro type in final front
        if self.preserve_diversity:
            for macro_name, (tree, fit) in macro_elite.items():
                opt_tree, _ = optimize_constants(tree, x, y)
                opt_fit = calculate_fitness(opt_tree, x, y, self.parsimony_coefficient)
                # Only add if it would be non-dominated or represents unique structure
                is_new_structure = not any(
                    self.get_macro_signature(t) == macro_name
                    for t, _ in optimized_front
                )
                if is_new_structure:
                    optimized_front.append((opt_tree, opt_fit))

        return optimized_front

    def _update_pareto_archive(
        self,
        archive: List[Tuple[ExprNode, FitnessResult]],
        tree: ExprNode,
        fit: FitnessResult
    ) -> None:
        """Update Pareto archive with new solution if non-dominated"""
        # Check if dominated by any archive member
        for _, archive_fit in archive:
            if archive_fit.dominates(fit):
                return  # Dominated, don't add

        # Check for duplicate (same complexity and similar MSE)
        for _, archive_fit in archive:
            if (archive_fit.complexity == fit.complexity and
                abs(archive_fit.mse - fit.mse) < 1e-10):
                return  # Duplicate, don't add

        # Remove any archive members dominated by new solution
        archive[:] = [
            (t, f) for t, f in archive
            if not fit.dominates(f)
        ]

        # Add new solution
        archive.append((tree.copy(), fit))

        # Limit archive size - keep best at each complexity level
        if len(archive) > 100:
            # Group by complexity and keep best MSE at each level
            by_complexity: dict = {}
            for t, f in archive:
                c = f.complexity
                if c not in by_complexity or f.mse < by_complexity[c][1].mse:
                    by_complexity[c] = (t, f)

            # Convert back to list, sorted by complexity
            archive[:] = sorted(by_complexity.values(), key=lambda x: x[1].complexity)

    def probe(
        self,
        x: np.ndarray,
        y: np.ndarray,
        generations: int = 5,
    ) -> FitnessResult:
        """
        Run a quick probe evolution to estimate how well this primitive set fits.

        Used by auto-discovery to compare different domains.

        Args:
            x: Input data
            y: Target data
            generations: Number of generations to run (default 5)

        Returns:
            Best FitnessResult achieved
        """
        # Initialize with pre-optimized macros for better domain detection
        population = self.initialize_population(x, y)
        best_fit = FitnessResult(
            mse=float('inf'),
            r_squared=-float('inf'),
            complexity=0,
            depth=0,
            score=-float('inf')
        )

        for _ in range(generations):
            fitness_scores = []
            for tree in population:
                fit = calculate_fitness(tree, x, y, self.parsimony_coefficient)
                fitness_scores.append(fit)
                if fit.r_squared > best_fit.r_squared:
                    best_fit = fit

            # Create next generation
            next_population = []

            # Elitism
            elite_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i].score,
                reverse=True
            )[:self.elitism]

            for idx in elite_indices:
                next_population.append(population[idx].copy())

            # Fill with offspring
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_prob:
                    p1 = self.tournament_select(population, fitness_scores)
                    p2 = self.tournament_select(population, fitness_scores)
                    c1, c2 = self.crossover(p1, p2)
                    if random.random() < self.mutation_prob:
                        c1 = self.mutate(c1)
                    next_population.append(c1)
                    if len(next_population) < self.population_size:
                        if random.random() < self.mutation_prob:
                            c2 = self.mutate(c2)
                        next_population.append(c2)
                else:
                    p = self.tournament_select(population, fitness_scores)
                    if random.random() < self.mutation_prob:
                        p = self.mutate(p)
                    next_population.append(p)

            population = next_population[:self.population_size]

        return best_fit


class SymbolicRegressor:
    """
    Symbolic Regression for discovering new functional forms.

    Uses genetic programming to search the space of mathematical
    expressions and find interpretable formulas that fit data.

    Example:
        # Basic usage with auto-discovery
        regressor = SymbolicRegressor()
        result = regressor.discover(x, y, mode=DiscoveryMode.AUTO, verbose=True)

        # Expert mode - specify domain
        result = regressor.discover(x, y, mode=DiscoveryMode.OSCILLATOR)

        # Pure discovery - no macros
        result = regressor.discover(x, y, mode=DiscoveryMode.DISCOVER)

        # Direct primitive set override
        result = regressor.discover(x, y, primitives=CIRCUIT_PRIMITIVES)

    Args:
        primitives: PrimitiveSet defining available operators (default for non-AUTO modes)
        population_size: Number of individuals per generation
        generations: Number of evolutionary generations
        max_depth: Maximum tree depth
        parsimony_coefficient: Complexity penalty (higher = simpler solutions)
        optimize_constants: Whether to run scipy optimization on constants
        random_state: Random seed for reproducibility
        probe_generations: Generations to run during domain probing (AUTO mode)
        residual_threshold: Threshold for detecting structured residuals
    """

    def __init__(
        self,
        primitives: Optional[PrimitiveSet] = None,
        population_size: int = 500,
        generations: int = 50,
        max_depth: int = 6,
        parsimony_coefficient: float = 0.001,
        optimize_constants: bool = True,
        random_state: Optional[int] = None,
        probe_generations: int = 5,
        residual_threshold: float = 0.15,
    ):
        if population_size < 10:
            raise ValueError("population_size must be at least 10")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if max_depth < 2:
            raise ValueError("max_depth must be at least 2")
        if parsimony_coefficient < 0:
            raise ValueError("parsimony_coefficient must be non-negative")

        self.primitives = primitives or PrimitiveSet()
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.optimize_constants = optimize_constants
        self.random_state = random_state
        self.probe_generations = probe_generations
        self.residual_threshold = residual_threshold

        self._engine = GPEngine(
            primitives=self.primitives,
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            parsimony_coefficient=parsimony_coefficient,
        )

    def discover(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mode: DiscoveryMode = DiscoveryMode.AUTO,
        primitives: Optional[PrimitiveSet] = None,
        verbose: bool = False
    ) -> SymbolicRegressionResult:
        """
        Discover symbolic expressions that fit the data.

        Args:
            x: Input data (1D array)
            y: Target data (1D array)
            mode: Discovery mode (AUTO, IDENTIFY, DISCOVER, or domain-specific)
            primitives: Expert override - bypasses mode selection
            verbose: Print progress during evolution

        Returns:
            SymbolicRegressionResult with Pareto front of solutions
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if len(x) < 5:
            raise ValueError("Need at least 5 data points")

        # Expert override takes precedence
        if primitives is not None:
            return self._direct_evolve(x, y, primitives, verbose)

        # Explicit mode selection
        if mode != DiscoveryMode.AUTO:
            prims = MODE_TO_PRIMITIVES.get(mode, self.primitives)
            return self._direct_evolve(x, y, prims, verbose)

        # AUTO mode: hierarchical search
        return self._auto_discover(x, y, verbose)

    def _direct_evolve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        primitives: PrimitiveSet,
        verbose: bool,
    ) -> SymbolicRegressionResult:
        """Standard evolution with specified primitives."""
        engine = GPEngine(
            primitives=primitives,
            population_size=self.population_size,
            generations=self.generations,
            max_depth=self.max_depth,
            parsimony_coefficient=self.parsimony_coefficient,
        )

        # Progress callback
        def progress_callback(gen: int, best_fit: FitnessResult):
            if verbose:
                print(f"  Gen {gen:3d}: R^2={best_fit.r_squared:.4f}, "
                      f"MSE={best_fit.mse:.4e}, Complexity={best_fit.complexity}")

        # Run evolution
        pareto_front = engine.evolve(
            x, y,
            callback=progress_callback if verbose else None
        )

        # Build result
        return self._build_result(pareto_front, x, y)

    def _auto_discover(
        self,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
    ) -> SymbolicRegressionResult:
        """Hierarchical auto-discovery with domain probing and residual analysis."""

        # Phase 1: Domain probe
        if verbose:
            print("Phase 1: Domain detection...")

        probe_results = self._probe_domains(x, y, verbose)
        best_domain, best_r2 = max(probe_results.items(), key=lambda kv: kv[1])

        if verbose:
            print(f"\n  Best domain: {best_domain} (R^2={best_r2:.3f})")
            for domain, r2 in sorted(probe_results.items(), key=lambda kv: -kv[1]):
                marker = " <--" if domain == best_domain else ""
                print(f"    {domain}: R^2={r2:.3f}{marker}")

        # Phase 2: Focused evolution
        if verbose:
            print(f"\nPhase 2: Focused evolution with {best_domain}...")

        primitives = dict(DOMAIN_PROBE_ORDER)[best_domain]
        main_result = self._direct_evolve(x, y, primitives, verbose)

        # Phase 3: Residual analysis
        if verbose:
            print("\nPhase 3: Residual analysis...")

        best_expr = main_result.most_accurate
        if best_expr is None:
            main_result.metadata["domain_detected"] = best_domain
            main_result.metadata["probe_results"] = probe_results
            return main_result

        y_pred = best_expr.expression.evaluate(x)
        residuals = y - y_pred

        # Check if residuals are structured
        residual_structure = self._measure_residual_structure(residuals)

        if verbose:
            print(f"  Residual structure: {residual_structure:.3f}")
            print(f"  Threshold: {self.residual_threshold}")

        # Store metadata
        main_result.metadata["domain_detected"] = best_domain
        main_result.metadata["probe_results"] = probe_results
        main_result.metadata["residual_structure"] = residual_structure
        main_result.metadata["correction_applied"] = False

        if residual_structure < self.residual_threshold:
            # Residuals are noise - we're done
            if verbose:
                print("  -> Residuals are noise. Done.")
            return main_result

        # Phase 4: Pure GP on residuals
        if verbose:
            print(f"  -> Residuals are structured. Applying pure GP...")

        correction_result = self._direct_evolve(
            x, residuals, MINIMAL_PRIMITIVES, verbose=False
        )
        correction_expr = correction_result.most_accurate

        # Only use correction if it meaningfully improves fit
        if correction_expr is not None and correction_expr.r_squared > 0.3:
            combined = self._combine_expressions(
                best_expr.expression,
                correction_expr.expression
            )
            combined_r2 = self._compute_r_squared(combined, x, y)

            if verbose:
                print(f"  Correction R^2 on residuals: {correction_expr.r_squared:.3f}")
                print(f"  Combined model R^2: {combined_r2:.3f}")

            if combined_r2 > best_expr.r_squared + 0.01:
                # Correction helps - update result
                main_result.metadata["correction_applied"] = True
                main_result.metadata["correction_expr"] = correction_expr.expression_string
                main_result.metadata["original_r2"] = best_expr.r_squared

                # Create combined fit result
                combined_residuals = y - combined.evaluate(x)
                combined_fit = SymbolicFitResult(
                    expression=combined,
                    expression_string=f"({best_expr.expression_string}) + ({correction_expr.expression_string})",
                    r_squared=combined_r2,
                    mse=float(np.mean(combined_residuals**2)),
                    complexity=best_expr.complexity + correction_expr.complexity,
                    residuals=combined_residuals,
                    residual_std=float(np.std(combined_residuals)),
                    is_noise_like=check_residuals_are_noise(combined_residuals),
                    params=np.concatenate([best_expr.params, correction_expr.params])
                )
                main_result.most_accurate = combined_fit

                if verbose:
                    print(f"  -> Combined model applied.")

        return main_result

    def _probe_domains(
        self,
        x: np.ndarray,
        y: np.ndarray,
        verbose: bool,
    ) -> dict:
        """Quick probe of each domain (few generations) to find best fit."""
        results = {}

        for domain_name, primitives in DOMAIN_PROBE_ORDER:
            engine = GPEngine(
                primitives=primitives,
                population_size=self.population_size // 2,  # Smaller for speed
                max_depth=self.max_depth,
                parsimony_coefficient=self.parsimony_coefficient,
            )

            best_fit = engine.probe(x, y, generations=self.probe_generations)
            results[domain_name] = best_fit.r_squared

            if verbose:
                print(f"    Probing {domain_name}... R^2={best_fit.r_squared:.3f}")

        return results

    def _measure_residual_structure(self, residuals: np.ndarray) -> float:
        """
        Measure how much structure remains in residuals.

        Returns 0 for pure noise, higher for structured residuals.
        Uses autocorrelation - noise has no autocorrelation,
        structured signals do.
        """
        # Normalize
        r = residuals - np.mean(residuals)
        if np.std(r) < 1e-10:
            return 0.0
        r = r / np.std(r)

        # Compute autocorrelation at lags 1-5
        n = len(r)
        autocorr = 0.0
        count = 0

        for lag in range(1, min(6, n // 4)):
            if lag < n:
                c = np.corrcoef(r[:-lag], r[lag:])[0, 1]
                if not np.isnan(c):
                    autocorr += abs(c)
                    count += 1

        return autocorr / max(count, 1)

    def _combine_expressions(
        self,
        main: ExprNode,
        correction: ExprNode,
    ) -> ExprNode:
        """Combine main expression with correction term via addition."""
        from .symbolic_types import BinaryOp
        return ExprNode(
            node_type=NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=main.copy(),
            right=correction.copy(),
        )

    def _compute_r_squared(
        self,
        tree: ExprNode,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute R-squared for a tree on data."""
        try:
            y_pred = tree.evaluate(x)
            if not np.all(np.isfinite(y_pred)):
                return -float('inf')
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        except Exception:
            return -float('inf')

    def _build_result(
        self,
        pareto_front: List[Tuple[ExprNode, FitnessResult]],
        x: np.ndarray,
        y: np.ndarray
    ) -> SymbolicRegressionResult:
        """Build SymbolicRegressionResult from Pareto front"""

        symbolic_results = []
        for tree, fit in pareto_front:
            y_pred = tree.evaluate(x)
            residuals = y - y_pred

            sfr = SymbolicFitResult(
                expression=tree,
                expression_string=tree.to_string(),
                r_squared=fit.r_squared,
                mse=fit.mse,
                complexity=fit.complexity,
                residuals=residuals,
                residual_std=float(np.std(residuals)),
                is_noise_like=check_residuals_are_noise(residuals),
                params=np.array(tree.get_constants())
            )
            symbolic_results.append(sfr)

        if not symbolic_results:
            return SymbolicRegressionResult(
                generations_run=self.generations,
                total_evaluations=self.population_size * self.generations
            )

        # Find best by different criteria
        most_accurate = max(symbolic_results, key=lambda r: r.r_squared)
        most_parsimonious = min(symbolic_results, key=lambda r: r.complexity)

        # Best tradeoff: knee of Pareto front
        best_tradeoff = self._find_knee(symbolic_results)

        return SymbolicRegressionResult(
            pareto_front=symbolic_results,
            most_accurate=most_accurate,
            most_parsimonious=most_parsimonious,
            best_tradeoff=best_tradeoff,
            generations_run=self.generations,
            total_evaluations=self.population_size * self.generations
        )

    def _find_knee(self, results: List[SymbolicFitResult]) -> SymbolicFitResult:
        """
        Find the knee of the Pareto front.

        Uses a weighted score that balances accuracy and simplicity,
        favoring solutions with good R² relative to their complexity.
        """
        if len(results) <= 1:
            return results[0] if results else None

        if len(results) == 2:
            return max(results, key=lambda r: r.r_squared)

        # Sort by complexity
        sorted_results = sorted(results, key=lambda r: r.complexity)

        # Normalize R² and complexity to [0, 1] range
        r_squared_values = [r.r_squared for r in sorted_results]
        complexity_values = [r.complexity for r in sorted_results]

        min_r2, max_r2 = min(r_squared_values), max(r_squared_values)
        min_c, max_c = min(complexity_values), max(complexity_values)

        # Avoid division by zero
        r2_range = max_r2 - min_r2 if max_r2 > min_r2 else 1.0
        c_range = max_c - min_c if max_c > min_c else 1.0

        # Find point with best normalized score
        # Score = normalized_r2 - 0.3 * normalized_complexity
        # This gives more weight to R² but still prefers simpler solutions
        best_score = -float('inf')
        knee = sorted_results[0]

        for r in sorted_results:
            norm_r2 = (r.r_squared - min_r2) / r2_range
            norm_c = (r.complexity - min_c) / c_range

            # Score that favors high R² with moderate complexity penalty
            score = norm_r2 - 0.3 * norm_c

            if score > best_score:
                best_score = score
                knee = r

        return knee
