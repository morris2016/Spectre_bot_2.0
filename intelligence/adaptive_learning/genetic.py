#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Genetic Algorithm Module for Strategy Evolution

This module implements advanced genetic algorithms for evolving trading strategies.
It provides a framework for strategy optimization through genetic operations like
crossover, mutation, and natural selection, enabling strategies to adapt to changing
market conditions.
"""

import os
import time
import uuid
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import json
import copy
from datetime import datetime, timedelta

# Internal imports
from common.logger import get_logger
from common.utils import (
    generate_id, serialize_dict, deserialize_dict, 
    safe_divide, get_timestamp, timestamp_to_datetime
)
from common.exceptions import (
    EvolutionError, InvalidPopulationError, 
    ConvergenceError, GeneticOperationError
)
from common.constants import (
    GENETIC_POPULATION_SIZE, GENETIC_GENERATIONS, 
    GENETIC_MUTATION_RATE, GENETIC_CROSSOVER_RATE,
    GENETIC_SELECTION_PRESSURE, GENETIC_ELITISM_RATE,
    GENETIC_TOURNAMENT_SIZE, MAX_THREADS, MIN_FITNESS
)
from data_storage.models.strategy_data import StrategyGeneticHistory
from data_storage.database import db_session
from backtester.engine import BacktestEngine

logger = get_logger("genetic_algorithm")

@dataclass
class Gene:
    """Represents a single parameter in a strategy."""
    name: str
    value: Any
    min_value: Any
    max_value: Any
    type: str
    mutation_std: float = 0.1
    step: Optional[float] = None
    discrete: bool = False
    options: Optional[List[Any]] = None
    
    def mutate(self, mutation_rate: float, custom_mutation_fn: Optional[Callable] = None) -> None:
        """Mutate this gene based on its type and constraints."""
        if random.random() > mutation_rate:
            return
        
        if custom_mutation_fn is not None:
            self.value = custom_mutation_fn(self)
            return
        
        try:
            if self.options is not None and len(self.options) > 0:
                self.value = random.choice(self.options)
                return
                
            if self.type == "int":
                if self.discrete and self.step is not None:
                    steps = int((self.max_value - self.min_value) / self.step)
                    current_step = int((self.value - self.min_value) / self.step)
                    new_step = max(0, min(steps, current_step + random.choice([-1, 0, 1])))
                    self.value = self.min_value + new_step * self.step
                else:
                    range_width = self.max_value - self.min_value
                    mutation_amount = int(random.gauss(0, range_width * self.mutation_std))
                    self.value = max(self.min_value, min(self.max_value, self.value + mutation_amount))
                    
            elif self.type == "float":
                if self.discrete and self.step is not None:
                    steps = int((self.max_value - self.min_value) / self.step)
                    current_step = int((self.value - self.min_value) / self.step)
                    new_step = max(0, min(steps, current_step + random.choice([-1, 0, 1])))
                    self.value = self.min_value + new_step * self.step
                else:
                    range_width = self.max_value - self.min_value
                    mutation_amount = random.gauss(0, range_width * self.mutation_std)
                    self.value = max(self.min_value, min(self.max_value, self.value + mutation_amount))
                    
            elif self.type == "bool":
                self.value = not self.value
                
            elif self.type == "category":
                if self.options is not None and len(self.options) > 0:
                    current_idx = self.options.index(self.value) if self.value in self.options else 0
                    shift = random.choice([-1, 1])
                    new_idx = (current_idx + shift) % len(self.options)
                    self.value = self.options[new_idx]
                    
            elif self.type == "string":
                if self.options is not None and len(self.options) > 0:
                    self.value = random.choice(self.options)
        except Exception as e:
            logger.error(f"Error mutating gene {self.name}: {str(e)}")
            # Don't change the value if mutation fails
    
    def crossover(self, other_gene: 'Gene', method: str = "uniform") -> Tuple['Gene', 'Gene']:
        """Perform crossover between this gene and another gene."""
        if self.type != other_gene.type:
            return copy.deepcopy(self), copy.deepcopy(other_gene)
        
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other_gene)
        
        try:
            if method == "uniform":
                if random.random() < 0.5:
                    child1.value, child2.value = child2.value, child1.value
                    
            elif method == "blend" and self.type in ["int", "float"]:
                # Blend crossover for numerical values
                beta = random.random()
                # Ensure values stay within bounds
                min_val = min(self.value, other_gene.value)
                max_val = max(self.value, other_gene.value)
                range_expansion = 0.1  # Allow some exploration beyond parents
                lower_bound = max(self.min_value, min_val - range_expansion * (max_val - min_val))
                upper_bound = min(self.max_value, max_val + range_expansion * (max_val - min_val))
                
                child1.value = lower_bound + beta * (upper_bound - lower_bound)
                child2.value = lower_bound + (1 - beta) * (upper_bound - lower_bound)
                
                if self.type == "int":
                    child1.value = int(child1.value)
                    child2.value = int(child2.value)
                
                if self.discrete and self.step is not None:
                    child1.value = round(child1.value / self.step) * self.step
                    child2.value = round(child2.value / self.step) * self.step
                    
            elif method == "simulated_binary" and self.type in ["int", "float"]:
                # Simulated Binary Crossover (SBX)
                eta = 2.0  # Distribution index
                
                if abs(self.value - other_gene.value) > 1e-10:  # Avoid division by zero
                    # Calculate gamma
                    if self.value < other_gene.value:
                        y1, y2 = self.value, other_gene.value
                    else:
                        y1, y2 = other_gene.value, self.value
                        
                    beta = 1.0 + (2.0 * (y1 - self.min_value) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    rand = random.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    # Generate children
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    # Ensure values are within bounds
                    c1 = max(self.min_value, min(self.max_value, c1))
                    c2 = max(self.min_value, min(self.max_value, c2))
                    
                    if self.type == "int":
                        c1 = int(c1)
                        c2 = int(c2)
                        
                    if self.discrete and self.step is not None:
                        c1 = round(c1 / self.step) * self.step
                        c2 = round(c2 / self.step) * self.step
                        
                    child1.value = c1
                    child2.value = c2
                    
        except Exception as e:
            logger.error(f"Error in crossover for gene {self.name}: {str(e)}")
            # Return copies of originals if crossover fails
            return copy.deepcopy(self), copy.deepcopy(other_gene)
            
        return child1, child2
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "type": self.type,
            "mutation_std": self.mutation_std,
            "step": self.step,
            "discrete": self.discrete,
            "options": self.options
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Gene':
        """Create gene from dictionary."""
        return cls(**data)


@dataclass
class Genome:
    """Represents a complete set of strategy parameters."""
    id: str = field(default_factory=lambda: generate_id("genome"))
    genes: Dict[str, Gene] = field(default_factory=dict)
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    evaluated: bool = False
    
    def mutate(self, mutation_rate: float, custom_mutations: Optional[Dict[str, Callable]] = None) -> None:
        """Mutate all genes in the genome."""
        for gene_name, gene in self.genes.items():
            custom_fn = None
            if custom_mutations is not None and gene_name in custom_mutations:
                custom_fn = custom_mutations[gene_name]
            gene.mutate(mutation_rate, custom_fn)
            
    def crossover(self, other_genome: 'Genome', crossover_rate: float = 0.7, 
                 method: str = "uniform") -> Tuple['Genome', 'Genome']:
        """Perform crossover with another genome."""
        if random.random() > crossover_rate:
            return copy.deepcopy(self), copy.deepcopy(other_genome)
            
        # Ensure both genomes have the same genes
        if set(self.genes.keys()) != set(other_genome.genes.keys()):
            logger.warning("Genomes have different gene sets, crossover may produce unexpected results")
            
        child1 = Genome(generation=self.generation + 1)
        child2 = Genome(generation=self.generation + 1)
        
        # Record parent IDs
        child1.parent_ids = [self.id, other_genome.id]
        child2.parent_ids = [self.id, other_genome.id]
        
        # Perform crossover for each gene
        common_genes = set(self.genes.keys()).intersection(set(other_genome.genes.keys()))
        
        for gene_name in common_genes:
            gene1, gene2 = self.genes[gene_name].crossover(other_genome.genes[gene_name], method)
            child1.genes[gene_name] = gene1
            child2.genes[gene_name] = gene2
            
        # Copy unique genes from each parent
        for gene_name in set(self.genes.keys()) - common_genes:
            child1.genes[gene_name] = copy.deepcopy(self.genes[gene_name])
            child2.genes[gene_name] = copy.deepcopy(self.genes[gene_name])
            
        for gene_name in set(other_genome.genes.keys()) - common_genes:
            child1.genes[gene_name] = copy.deepcopy(other_genome.genes[gene_name])
            child2.genes[gene_name] = copy.deepcopy(other_genome.genes[gene_name])
            
        return child1, child2
    
    def get_parameters(self) -> Dict[str, Any]:
        """Extract parameter values from genes."""
        return {name: gene.value for name, gene in self.genes.items()}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary."""
        return {
            "id": self.id,
            "genes": {name: gene.to_dict() for name, gene in self.genes.items()},
            "fitness": self.fitness,
            "metrics": self.metrics,
            "creation_time": self.creation_time,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "evaluated": self.evaluated
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Create genome from dictionary."""
        genes = {name: Gene.from_dict(gene_data) for name, gene_data in data["genes"].items()}
        genome = cls(
            id=data["id"],
            genes=genes,
            fitness=data["fitness"],
            metrics=data.get("metrics", {}),
            creation_time=data["creation_time"],
            parent_ids=data["parent_ids"],
            generation=data["generation"],
            evaluated=data.get("evaluated", False)
        )
        return genome
        
    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any], 
                       parameter_specs: Dict[str, Dict[str, Any]]) -> 'Genome':
        """Create a genome from parameter values and specifications."""
        genome = cls()
        for name, value in parameters.items():
            if name in parameter_specs:
                spec = parameter_specs[name]
                gene = Gene(
                    name=name,
                    value=value,
                    min_value=spec.get("min_value"),
                    max_value=spec.get("max_value"),
                    type=spec.get("type", "float"),
                    mutation_std=spec.get("mutation_std", 0.1),
                    step=spec.get("step"),
                    discrete=spec.get("discrete", False),
                    options=spec.get("options")
                )
                genome.genes[name] = gene
        return genome


class GeneticAlgorithm:
    """Advanced genetic algorithm for strategy evolution."""
    
    def __init__(
        self,
        parameter_specs: Dict[str, Dict[str, Any]],
        fitness_function: Callable[[Dict[str, Any]], Tuple[float, Dict[str, float]]],
        population_size: int = GENETIC_POPULATION_SIZE,
        generations: int = GENETIC_GENERATIONS,
        mutation_rate: float = GENETIC_MUTATION_RATE,
        crossover_rate: float = GENETIC_CROSSOVER_RATE,
        selection_pressure: float = GENETIC_SELECTION_PRESSURE,
        elitism_rate: float = GENETIC_ELITISM_RATE,
        tournament_size: int = GENETIC_TOURNAMENT_SIZE,
        strategy_id: Optional[str] = None,
        asset: Optional[str] = None,
        platform: Optional[str] = None,
        timeframe: Optional[str] = None,
        custom_mutations: Optional[Dict[str, Callable]] = None,
        crossover_method: str = "uniform",
        parallel: bool = True,
        max_workers: int = MAX_THREADS,
        early_stopping_generations: int = 10,
        early_stopping_threshold: float = 0.001,
        save_history: bool = True
    ):
        """Initialize the genetic algorithm."""
        self.parameter_specs = parameter_specs
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.strategy_id = strategy_id
        self.asset = asset
        self.platform = platform
        self.timeframe = timeframe
        self.custom_mutations = custom_mutations
        self.crossover_method = crossover_method
        self.parallel = parallel
        self.max_workers = max_workers
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_threshold = early_stopping_threshold
        self.save_history = save_history
        self.current_generation = 0
        
        # Init population
        self.population: List[Genome] = []
        self.best_genomes: List[Genome] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
    def initialize_population(self, initial_parameters: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize the population with random or provided parameters."""
        self.population = []
        
        if initial_parameters is not None:
            # Use provided parameters for initial population
            for params in initial_parameters:
                genome = Genome.from_parameters(params, self.parameter_specs)
                self.population.append(genome)
                
            # If provided parameters are fewer than population size, generate random ones
            remaining = self.population_size - len(initial_parameters)
            if remaining > 0:
                self._generate_random_genomes(remaining)
        else:
            # Generate completely random population
            self._generate_random_genomes(self.population_size)
            
        self.current_generation = 0
            
    def _generate_random_genomes(self, count: int) -> None:
        """Generate random genomes within parameter constraints."""
        for _ in range(count):
            genome = Genome()
            
            for name, spec in self.parameter_specs.items():
                gene_type = spec.get("type", "float")
                min_val = spec.get("min_value")
                max_val = spec.get("max_value")
                options = spec.get("options")
                discrete = spec.get("discrete", False)
                step = spec.get("step")
                
                # Generate random value based on type
                value = None
                if options is not None and len(options) > 0:
                    value = random.choice(options)
                elif gene_type == "int":
                    if discrete and step is not None:
                        steps = int((max_val - min_val) / step)
                        step_idx = random.randint(0, steps)
                        value = min_val + step_idx * step
                    else:
                        value = random.randint(min_val, max_val)
                elif gene_type == "float":
                    if discrete and step is not None:
                        steps = int((max_val - min_val) / step)
                        step_idx = random.randint(0, steps)
                        value = min_val + step_idx * step
                    else:
                        value = min_val + random.random() * (max_val - min_val)
                elif gene_type == "bool":
                    value = random.choice([True, False])
                elif gene_type == "category" and options is not None:
                    value = random.choice(options)
                elif gene_type == "string" and options is not None:
                    value = random.choice(options)
                
                # Create and add gene
                gene = Gene(
                    name=name,
                    value=value,
                    min_value=min_val,
                    max_value=max_val,
                    type=gene_type,
                    mutation_std=spec.get("mutation_std", 0.1),
                    step=step,
                    discrete=discrete,
                    options=options
                )
                genome.genes[name] = gene
                
            self.population.append(genome)
            
    def evaluate_fitness(self) -> None:
        """Evaluate fitness for all genomes in the population."""
        if self.parallel and self.max_workers > 1:
            self._evaluate_fitness_parallel()
        else:
            self._evaluate_fitness_sequential()
            
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best genome if needed
        if (not self.best_genomes or 
            self.population[0].fitness > self.best_genomes[-1].fitness):
            self.best_genomes.append(copy.deepcopy(self.population[0]))
            
        # Record generation statistics
        self._record_generation_stats()
            
    def _evaluate_fitness_sequential(self) -> None:
        """Evaluate fitness for all genomes sequentially."""
        for genome in self.population:
            if not genome.evaluated:
                parameters = genome.get_parameters()
                try:
                    fitness, metrics = self.fitness_function(parameters)
                    with self._lock:
                        genome.fitness = fitness
                        genome.metrics = metrics
                        genome.evaluated = True
                except Exception as e:
                    logger.error(f"Error evaluating genome {genome.id}: {str(e)}")
                    genome.fitness = MIN_FITNESS
                    genome.evaluated = True
                    
    def _evaluate_fitness_parallel(self) -> None:
        """Evaluate fitness for all genomes in parallel."""
        unevaluated_genomes = [g for g in self.population if not g.evaluated]
        parameter_sets = [g.get_parameters() for g in unevaluated_genomes]
        genome_ids = [g.id for g in unevaluated_genomes]
        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._fitness_wrapper, params, gid) 
                      for params, gid in zip(parameter_sets, genome_ids)]
            
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in parallel fitness evaluation: {str(e)}")
                    results.append((MIN_FITNESS, {}, None))
                    
        # Update genomes with results
        for genome, (fitness, metrics, _) in zip(unevaluated_genomes, results):
            with self._lock:
                genome.fitness = fitness
                genome.metrics = metrics
                genome.evaluated = True
                
    def _fitness_wrapper(self, parameters: Dict[str, Any], genome_id: str) -> Tuple[float, Dict[str, float], str]:
        """Wrapper for fitness function to catch exceptions in parallel execution."""
        try:
            fitness, metrics = self.fitness_function(parameters)
            return fitness, metrics, genome_id
        except Exception as e:
            logger.error(f"Error evaluating genome {genome_id}: {str(e)}")
            return MIN_FITNESS, {}, genome_id
                
    def _record_generation_stats(self) -> None:
        """Record statistics for the current generation."""
        if not self.population:
            return
            
        fitnesses = [g.fitness for g in self.population]
        
        stats = {
            "generation": self.current_generation,
            "timestamp": time.time(),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "median_fitness": sorted(fitnesses)[len(fitnesses) // 2],
            "std_fitness": np.std(fitnesses),
            "population_size": len(self.population),
            "best_genome_id": self.population[0].id if self.population else None,
            "best_parameters": self.population[0].get_parameters() if self.population else {},
            "best_metrics": self.population[0].metrics if self.population else {}
        }
        
        self.evolution_history.append(stats)
        
        # Save to database if enabled
        if self.save_history and self.strategy_id:
            try:
                with db_session() as session:
                    history_entry = StrategyGeneticHistory(
                        strategy_id=self.strategy_id,
                        generation=self.current_generation,
                        timestamp=datetime.fromtimestamp(time.time()),
                        asset=self.asset,
                        platform=self.platform,
                        timeframe=self.timeframe,
                        best_fitness=stats["best_fitness"],
                        avg_fitness=stats["avg_fitness"],
                        population_size=len(self.population),
                        best_parameters=json.dumps(stats["best_parameters"]),
                        best_metrics=json.dumps(stats["best_metrics"])
                    )
                    session.add(history_entry)
                    session.commit()
            except Exception as e:
                logger.error(f"Error saving evolution history: {str(e)}")
            
    def select_parents(self) -> List[Genome]:
        """Select parents for next generation using tournament selection."""
        num_parents = self.population_size - int(self.population_size * self.elitism_rate)
        parents = []
        
        for _ in range(num_parents):
            # Tournament selection
            tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
            tournament.sort(key=lambda x: x.fitness, reverse=True)
            
            # Apply selection pressure - higher pressure means more likely to select the best
            selection_idx = int(random.random() ** self.selection_pressure * len(tournament))
            parents.append(tournament[selection_idx])
            
        return parents
        
    def create_next_generation(self) -> None:
        """Create the next generation through crossover and mutation."""
        # Keep elite individuals
        elite_count = int(self.population_size * self.elitism_rate)
        elites = self.population[:elite_count]
        
        # Select parents for breeding
        parents = self.select_parents()
        
        # Create offspring through crossover and mutation
        offspring = []
        
        # Ensure even number of parents
        if len(parents) % 2 == 1:
            parents.append(random.choice(parents))
            
        # Perform crossover
        random.shuffle(parents)  # Randomize pairing
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = parents[i].crossover(
                    parents[i+1], 
                    self.crossover_rate, 
                    self.crossover_method
                )
                
                # Mutate children
                child1.mutate(self.mutation_rate, self.custom_mutations)
                child2.mutate(self.mutation_rate, self.custom_mutations)
                
                # Set generation
                child1.generation = self.current_generation + 1
                child2.generation = self.current_generation + 1
                
                offspring.extend([child1, child2])
                
        # Create new population
        new_population = []
        
        # Add elites (unchanged)
        for elite in elites:
            elite_copy = copy.deepcopy(elite)
            elite_copy.generation = self.current_generation + 1
            new_population.append(elite_copy)
            
        # Fill remaining population with offspring
        remaining_slots = self.population_size - len(new_population)
        new_population.extend(offspring[:remaining_slots])
        
        # If we don't have enough offspring, add random genomes
        if len(new_population) < self.population_size:
            missing = self.population_size - len(new_population)
            self._generate_random_genomes(missing)
            for genome in self.population[-missing:]:
                genome.generation = self.current_generation + 1
                new_population.append(genome)
                
        self.population = new_population
        self.current_generation += 1
        
    def evolve(self) -> Dict[str, Any]:
        """Run the full evolutionary process."""
        if not self.population:
            raise InvalidPopulationError("Population is not initialized")
            
        best_fitness_history = []
        convergence_counter = 0
        
        try:
            # Evaluate initial population
            self.evaluate_fitness()
            best_fitness_history.append(self.population[0].fitness)
            
            logger.info(f"Generation 0: Best fitness = {self.population[0].fitness}")
            
            # Evolution loop
            for generation in range(1, self.generations + 1):
                self.create_next_generation()
                self.evaluate_fitness()
                
                best_fitness = self.population[0].fitness
                best_fitness_history.append(best_fitness)
                
                logger.info(f"Generation {generation}: Best fitness = {best_fitness}")
                
                # Check for early stopping
                if len(best_fitness_history) > self.early_stopping_generations:
                    recent_best = best_fitness_history[-self.early_stopping_generations:]
                    improvement = (recent_best[-1] - recent_best[0]) / abs(recent_best[0]) if recent_best[0] != 0 else 0
                    
                    if abs(improvement) < self.early_stopping_threshold:
                        convergence_counter += 1
                    else:
                        convergence_counter = 0
                        
                    if convergence_counter >= 3:  # Require multiple generations with no improvement
                        logger.info(f"Early stopping at generation {generation}: No significant improvement")
                        break
                        
        except Exception as e:
            logger.error(f"Error during evolution: {str(e)}")
            raise EvolutionError(f"Evolution process failed: {str(e)}")
            
        # Return results
        best_genome = self.population[0] if self.population else None
        
        results = {
            "best_parameters": best_genome.get_parameters() if best_genome else {},
            "best_fitness": best_genome.fitness if best_genome else MIN_FITNESS,
            "best_metrics": best_genome.metrics if best_genome else {},
            "generations_completed": self.current_generation,
            "evolution_history": self.evolution_history,
            "population_size": len(self.population),
            "all_best_genomes": [g.to_dict() for g in self.best_genomes]
        }
        
        return results
        
    def get_best_genome(self) -> Optional[Genome]:
        """Get the best genome from the current population."""
        if not self.population:
            return None
            
        return max(self.population, key=lambda x: x.fitness)
        
    def add_existing_genome(self, parameters: Dict[str, Any], 
                          fitness: Optional[float] = None, 
                          metrics: Optional[Dict[str, float]] = None) -> None:
        """Add an existing set of parameters to the population."""
        genome = Genome.from_parameters(parameters, self.parameter_specs)
        
        if fitness is not None:
            genome.fitness = fitness
            genome.evaluated = True
            
        if metrics is not None:
            genome.metrics = metrics
            
        self.population.append(genome)
        
    def save_population(self, filepath: str) -> None:
        """Save the current population to a file."""
        try:
            data = {
                "population": [g.to_dict() for g in self.population],
                "best_genomes": [g.to_dict() for g in self.best_genomes],
                "current_generation": self.current_generation,
                "evolution_history": self.evolution_history,
                "parameter_specs": self.parameter_specs,
                "config": {
                    "population_size": self.population_size,
                    "generations": self.generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "selection_pressure": self.selection_pressure,
                    "elitism_rate": self.elitism_rate,
                    "tournament_size": self.tournament_size,
                    "strategy_id": self.strategy_id,
                    "asset": self.asset,
                    "platform": self.platform,
                    "timeframe": self.timeframe,
                    "crossover_method": self.crossover_method
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Population saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving population: {str(e)}")
            
    @classmethod
    def load_population(cls, filepath: str, fitness_function: Callable) -> 'GeneticAlgorithm':
        """Load a population from a file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Create instance with loaded config
            config = data.get("config", {})
            parameter_specs = data.get("parameter_specs", {})
            
            instance = cls(
                parameter_specs=parameter_specs,
                fitness_function=fitness_function,
                population_size=config.get("population_size", GENETIC_POPULATION_SIZE),
                generations=config.get("generations", GENETIC_GENERATIONS),
                mutation_rate=config.get("mutation_rate", GENETIC_MUTATION_RATE),
                crossover_rate=config.get("crossover_rate", GENETIC_CROSSOVER_RATE),
                selection_pressure=config.get("selection_pressure", GENETIC_SELECTION_PRESSURE),
                elitism_rate=config.get("elitism_rate", GENETIC_ELITISM_RATE),
                tournament_size=config.get("tournament_size", GENETIC_TOURNAMENT_SIZE),
                strategy_id=config.get("strategy_id"),
                asset=config.get("asset"),
                platform=config.get("platform"),
                timeframe=config.get("timeframe"),
                crossover_method=config.get("crossover_method", "uniform")
            )
            
            # Load population
            instance.population = [Genome.from_dict(g) for g in data.get("population", [])]
            instance.best_genomes = [Genome.from_dict(g) for g in data.get("best_genomes", [])]
            instance.current_generation = data.get("current_generation", 0)
            instance.evolution_history = data.get("evolution_history", [])
            
            logger.info(f"Population loaded from {filepath}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading population: {str(e)}")
            raise InvalidPopulationError(f"Failed to load population: {str(e)}")
            
    def inject_parameters(self, parameters_list: List[Dict[str, Any]], 
                        evaluate: bool = True) -> None:
        """Inject new parameter sets into the population."""
        for params in parameters_list:
            genome = Genome.from_parameters(params, self.parameter_specs)
            genome.generation = self.current_generation
            self.population.append(genome)
            
        # Evaluate if requested
        if evaluate:
            self.evaluate_fitness()
            
        # Trim population if it's grown too large
        if len(self.population) > self.population_size:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.population = self.population[:self.population_size]
            
    def plot_evolution(self, filepath: Optional[str] = None) -> None:
        """Plot the evolution history."""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            if not self.evolution_history:
                logger.warning("No evolution history to plot")
                return
                
            df = pd.DataFrame(self.evolution_history)
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(df["generation"], df["best_fitness"], label="Best Fitness", marker="o")
            plt.plot(df["generation"], df["avg_fitness"], label="Average Fitness", marker="x")
            plt.fill_between(df["generation"], 
                            df["best_fitness"], 
                            df["worst_fitness"], 
                            alpha=0.2)
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title("Evolution of Fitness")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(df["generation"], df["std_fitness"], label="Fitness Std Dev", marker="s")
            plt.xlabel("Generation")
            plt.ylabel("Standard Deviation")
            plt.title("Population Diversity")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            if filepath:
                plt.savefig(filepath)
                logger.info(f"Evolution plot saved to {filepath}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib or pandas not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting evolution: {str(e)}")


# Factory method for creating genetic algorithms for specific strategies
def create_genetic_optimizer(
    strategy_type: str,
    asset: str,
    platform: str,
    timeframe: str,
    custom_params: Optional[Dict[str, Any]] = None
) -> GeneticAlgorithm:
    """Create a genetic algorithm optimizer for a specific strategy type."""
    from backtester.engine import BacktestEngine
    from strategy_brains.base_brain import get_strategy_parameter_specs
    
    # Get parameter specifications for this strategy type
    param_specs = get_strategy_parameter_specs(strategy_type)
    
    # Create backtester
    backtester = BacktestEngine()
    
    # Define fitness function that uses backtester
    def fitness_function(parameters: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        try:
            # Configure and run backtest
            results = backtester.run_backtest(
                strategy_type=strategy_type,
                parameters=parameters,
                asset=asset,
                platform=platform,
                timeframe=timeframe,
                start_date=datetime.now() - timedelta(days=30),  # Last 30 days
                end_date=datetime.now(),
                initial_capital=1000.0
            )
            
            # Extract key metrics
            sharpe = results.get("sharpe_ratio", 0.0)
            profit_factor = results.get("profit_factor", 0.0)
            win_rate = results.get("win_rate", 0.0)
            max_drawdown = results.get("max_drawdown_pct", 100.0)
            total_trades = results.get("total_trades", 0)
            
            # Create composite fitness score
            # Higher is better, penalize low trade count and high drawdown
            fitness = 0.0
            
            if total_trades >= 10:  # Require minimum number of trades
                # Balance between returns and risk
                fitness = (0.4 * sharpe + 
                          0.3 * profit_factor + 
                          0.2 * win_rate - 
                          0.1 * (max_drawdown / 20.0))  # Normalize drawdown penalty
            else:
                # Penalize strategies with too few trades
                fitness = -1.0 + (total_trades / 10.0)  # Gradually less negative as trades increase
                
            # Create metrics dictionary
            metrics = {
                "sharpe_ratio": sharpe,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "max_drawdown_pct": max_drawdown,
                "total_trades": total_trades,
                "total_return_pct": results.get("total_return_pct", 0.0),
                "avg_trade_duration": results.get("avg_trade_duration", 0.0)
            }
            
            return fitness, metrics
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {str(e)}")
            return MIN_FITNESS, {}
    
    # Configure genetic algorithm
    config = {
        "parameter_specs": param_specs,
        "fitness_function": fitness_function,
        "strategy_id": f"{strategy_type}_{asset}_{timeframe}",
        "asset": asset,
        "platform": platform,
        "timeframe": timeframe
    }
    
    # Apply any custom parameters
    if custom_params:
        config.update(custom_params)
        
    # Create and return optimizer
    return GeneticAlgorithm(**config)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Strategy Optimization")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy type to optimize")
    parser.add_argument("--asset", type=str, required=True, help="Asset to trade")
    parser.add_argument("--platform", type=str, required=True, help="Trading platform")
    parser.add_argument("--timeframe", type=str, required=True, help="Trading timeframe")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--output", type=str, help="Output file for best parameters")
    
    args = parser.parse_args()
    
    try:
        # Create genetic optimizer
        optimizer = create_genetic_optimizer(
            strategy_type=args.strategy,
            asset=args.asset,
            platform=args.platform,
            timeframe=args.timeframe,
            custom_params={
                "population_size": args.pop_size,
                "generations": args.generations
            }
        )
        
        # Initialize and evolve
        optimizer.initialize_population()
        results = optimizer.evolve()
        
        # Print results
        print("\nEvolution Results:")
        print(f"Best Fitness: {results['best_fitness']}")
        print(f"Generations: {results['generations_completed']}")
        print("Best Parameters:")
        for param, value in results['best_parameters'].items():
            print(f"  {param}: {value}")
        print("\nMetrics:")
        for metric, value in results['best_metrics'].items():
            print(f"  {metric}: {value}")
            
        # Save results if output specified
        if args.output:
            optimizer.save_population(args.output)
            optimizer.plot_evolution(args.output.replace(".json", ".png"))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

