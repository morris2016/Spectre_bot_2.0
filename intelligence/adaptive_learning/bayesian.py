#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Bayesian Optimization Module

This module implements Bayesian optimization techniques for hyperparameter tuning
and strategy optimization. It uses Gaussian Processes to model the objective function
and efficiently explore the parameter space to find optimal configurations.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from dataclasses import dataclass, field
import concurrent.futures
from functools import partial
import json

# Scientific libraries
import scipy
from scipy.stats import norm
from scipy.optimize import minimize

# ML Libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
import joblib

# Internal imports
from common.logger import get_logger
from common.utils import create_directory, get_timestamp
from common.constants import (
    BAYESIAN_OPT_KERNELS, DEFAULT_GP_KERNEL, ACQUISITION_FUNCTIONS,
    DEFAULT_ACQUISITION, OPTIMIZATION_DIRECTION, DEFAULT_PARAM_BOUNDS,
    MAX_PARALLEL_EVALUATIONS, GP_RANDOM_RESTARTS
)
from common.metrics import compute_sharpe_ratio, compute_sortino_ratio
from common.exceptions import OptimizationError, ModelTrainingError

logger = get_logger("intelligence.adaptive_learning.bayesian")


@dataclass
class OptimizationResult:
    """Class to hold optimization results"""
    best_params: Dict[str, float]
    best_value: float
    all_params: List[Dict[str, float]]
    all_values: List[float]
    acquisition_values: List[float] = field(default_factory=list)
    optimization_time: float = 0.0
    iterations: int = 0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "best_params": self.best_params,
            "best_value": float(self.best_value),
            "optimization_time": self.optimization_time,
            "iterations": self.iterations,
            "success": self.success,
            "error_message": self.error_message
        }
    
    def save(self, filepath: str) -> None:
        """Save optimization results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """Load optimization results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        result = cls(
            best_params=data["best_params"],
            best_value=data["best_value"],
            all_params=[],  # These are not saved in the file
            all_values=[],
            optimization_time=data["optimization_time"],
            iterations=data["iterations"],
            success=data["success"],
            error_message=data["error_message"]
        )
        return result


class BayesianOptimizer:
    """
    Advanced Bayesian Optimization for hyperparameter tuning and strategy optimization.
    Uses Gaussian Processes to model the objective function and efficiently explores
    the parameter space to find optimal configurations.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        optimization_direction: str = "maximize",
        kernel_type: str = DEFAULT_GP_KERNEL,
        acquisition_function: str = DEFAULT_ACQUISITION,
        initial_points: int = 5,
        random_state: int = 42,
        exploration_weight: float = 0.1,
        gp_params: Optional[Dict] = None,
        minimize_constraints: Optional[List] = None,
        verbose: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5,
        parallel_evaluations: int = 1
    ):
        """
        Initialize the Bayesian Optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (lower, upper) bounds
            objective_function: Function to optimize, should accept parameters as kwargs
            optimization_direction: Either "maximize" or "minimize"
            kernel_type: Type of kernel for the Gaussian Process
            acquisition_function: Acquisition function to use
            initial_points: Number of initial random evaluations
            random_state: Random seed for reproducibility
            exploration_weight: Weight for exploration vs exploitation
            gp_params: Additional parameters for Gaussian Process
            minimize_constraints: Constraints for the scipy.optimize.minimize function
            verbose: Whether to print verbose output
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: How often to save checkpoints (iterations)
            parallel_evaluations: Number of parallel evaluations to perform
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.bounds = np.array([param_bounds[name] for name in self.param_names])
        self.dim = len(param_bounds)
        
        # Check if bounds are valid
        for name, (lower, upper) in param_bounds.items():
            if lower >= upper:
                raise ValueError(f"Lower bound must be less than upper bound for parameter {name}")
        
        self.objective_function = objective_function
        self.optimization_direction = optimization_direction
        if optimization_direction not in OPTIMIZATION_DIRECTION:
            raise ValueError(f"optimization_direction must be one of {OPTIMIZATION_DIRECTION}")
        
        # Set up sign for optimization direction
        self.sign = -1 if optimization_direction == "maximize" else 1
        
        # Acquisition function setup
        self.acquisition_function = acquisition_function
        if acquisition_function not in ACQUISITION_FUNCTIONS:
            raise ValueError(f"acquisition_function must be one of {ACQUISITION_FUNCTIONS}")
        
        self.exploration_weight = exploration_weight
        self.initial_points = initial_points
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Setup Gaussian Process
        self.kernel = self._get_kernel(kernel_type)
        default_gp_params = {
            "alpha": 1e-6,
            "normalize_y": True,
            "n_restarts_optimizer": GP_RANDOM_RESTARTS,
            "random_state": random_state
        }
        if gp_params:
            default_gp_params.update(gp_params)
        
        self.gp = GaussianProcessRegressor(kernel=self.kernel, **default_gp_params)
        
        # Store optimization history
        self.X_samples = []
        self.y_samples = []
        self.y_raw = []
        
        # For parallel evaluation
        self.parallel_evaluations = min(parallel_evaluations, MAX_PARALLEL_EVALUATIONS)
        
        # Constraints for parameter optimization
        self.minimize_constraints = minimize_constraints
        
        # Setup for checkpointing
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            create_directory(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        
        logger.info(f"Initialized Bayesian Optimizer with {self.dim} parameters")
        logger.info(f"Optimization direction: {optimization_direction}")
        logger.info(f"Kernel type: {kernel_type}")
        logger.info(f"Acquisition function: {acquisition_function}")
        logger.info(f"Parallel evaluations: {self.parallel_evaluations}")
    
    def _get_kernel(self, kernel_type: str):
        """
        Create a kernel for the Gaussian Process.
        
        Args:
            kernel_type: Type of kernel to use
            
        Returns:
            The configured kernel
        """
        if kernel_type == "RBF":
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        elif kernel_type == "Matern":
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        elif kernel_type == "RBF+Matern":
            return (ConstantKernel(1.0) * RBF(length_scale=1.0) + 
                   ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) +
                   WhiteKernel(noise_level=0.1))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters dictionary to numpy array in correct order"""
        return np.array([params[name] for name in self.param_names])
    
    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameters dictionary"""
        return {name: float(val) for name, val in zip(self.param_names, x)}
    
    def _normalize_params(self, x: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range"""
        bounds = np.array(self.bounds)
        return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    def _denormalize_params(self, x: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] range to original range"""
        bounds = np.array(self.bounds)
        return x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    def _acquisition_ei(self, x: np.ndarray, gp, y_best: float) -> float:
        """
        Expected Improvement acquisition function.
        
        Args:
            x: The point to evaluate
            gp: The Gaussian Process model
            y_best: The best observed value
            
        Returns:
            The negative expected improvement (for minimization)
        """
        x = x.reshape(1, -1)
        
        # Get mean and standard deviation at x
        mu, sigma = gp.predict(x, return_std=True)
        
        # If standard deviation is zero, return zero improvement
        if sigma == 0:
            return 0
        
        # Calculate improvement based on optimization direction
        if self.optimization_direction == "maximize":
            z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization
        else:
            z = (y_best - mu) / sigma
            ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization
    
    def _acquisition_ucb(self, x: np.ndarray, gp, **kwargs) -> float:
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            x: The point to evaluate
            gp: The Gaussian Process model
            
        Returns:
            The negative UCB (for minimization)
        """
        x = x.reshape(1, -1)
        
        # Get mean and standard deviation at x
        mu, sigma = gp.predict(x, return_std=True)
        
        # Calculate UCB based on optimization direction
        if self.optimization_direction == "maximize":
            ucb = mu + self.exploration_weight * sigma
            return -ucb  # Negative for minimization
        else:
            ucb = mu - self.exploration_weight * sigma
            return ucb
    
    def _acquisition_poi(self, x: np.ndarray, gp, y_best: float) -> float:
        """
        Probability of Improvement acquisition function.
        
        Args:
            x: The point to evaluate
            gp: The Gaussian Process model
            y_best: The best observed value
            
        Returns:
            The negative probability of improvement (for minimization)
        """
        x = x.reshape(1, -1)
        
        # Get mean and standard deviation at x
        mu, sigma = gp.predict(x, return_std=True)
        
        # If standard deviation is zero, return zero probability
        if sigma == 0:
            return 0
        
        # Calculate improvement based on optimization direction
        if self.optimization_direction == "maximize":
            z = (mu - y_best) / sigma
            poi = norm.cdf(z)
            return -poi  # Negative for minimization
        else:
            z = (y_best - mu) / sigma
            poi = norm.cdf(z)
            return -poi  # Negative for minimization
    
    def _get_acquisition_function(self, y_best: float = None):
        """Get the appropriate acquisition function"""
        if self.acquisition_function == "EI":
            return partial(self._acquisition_ei, gp=self.gp, y_best=y_best)
        elif self.acquisition_function == "UCB":
            return partial(self._acquisition_ucb, gp=self.gp)
        elif self.acquisition_function == "POI":
            return partial(self._acquisition_poi, gp=self.gp, y_best=y_best)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def _optimize_acquisition(self, acquisition_function, n_restarts: int = 10) -> Tuple[np.ndarray, float]:
        """
        Optimize the acquisition function to find the next point to evaluate.
        
        Args:
            acquisition_function: The acquisition function to optimize
            n_restarts: Number of restarts for the optimization
            
        Returns:
            The best point found and its acquisition value
        """
        best_x = None
        best_acquisition_value = 1.0  # We're minimizing, so start with a high value
        
        bounds = [(0, 1)] * self.dim  # Normalized bounds
        
        # Try different starting points
        for _ in range(n_restarts):
            # Start from a random point
            x0 = np.random.rand(self.dim)
            
            # Optimize from this starting point
            result = minimize(
                fun=acquisition_function,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
                constraints=self.minimize_constraints
            )
            
            # Update best if this is better
            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x
        
        # Denormalize the result
        denormalized_x = self._denormalize_params(best_x)
        
        return denormalized_x, -best_acquisition_value  # Invert sign back
    
    def _evaluate_function(self, params: Dict[str, float]) -> float:
        """
        Evaluate the objective function with the given parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            The result of the objective function
        """
        try:
            result = self.objective_function(**params)
            # Apply sign based on optimization direction
            return self.sign * result, result
        except Exception as e:
            logger.error(f"Error evaluating function with parameters {params}: {str(e)}")
            # Return a very poor value to discourage this region of parameter space
            return self.sign * float('-inf' if self.optimization_direction == "maximize" else 'inf'), None
    
    def _evaluate_initial_points(self):
        """Evaluate initial random points to build initial model"""
        logger.info(f"Evaluating {self.initial_points} initial random points...")
        
        # Generate Latin Hypercube samples for better coverage
        from sklearn.utils.random import sample_without_replacement
        
        # Generate normalized points in [0, 1] range
        normalized_points = np.random.uniform(0, 1, size=(self.initial_points, self.dim))
        
        # Denormalize to actual parameter ranges
        points = []
        for i in range(self.initial_points):
            point = self._denormalize_params(normalized_points[i])
            points.append(self._array_to_params(point))
        
        # Evaluate in parallel if configured
        if self.parallel_evaluations > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_evaluations) as executor:
                results = list(executor.map(self._evaluate_function, points))
        else:
            results = [self._evaluate_function(p) for p in points]
        
        # Store results
        for i, (signed_result, raw_result) in enumerate(results):
            if raw_result is not None:  # Only add valid results
                self.X_samples.append(self._params_to_array(points[i]))
                self.y_samples.append(signed_result)
                self.y_raw.append(raw_result)
        
        logger.info(f"Initial evaluation complete. {len(self.X_samples)} valid points.")
    
    def _save_checkpoint(self, iteration: int, best_params: Dict, best_value: float):
        """Save a checkpoint of the optimization state"""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            "iteration": iteration,
            "best_params": best_params,
            "best_value": best_value,
            "X_samples": np.array(self.X_samples).tolist(),
            "y_samples": np.array(self.y_samples).tolist(),
            "y_raw": self.y_raw,
            "random_state": self.random_state,
            "timestamp": get_timestamp()
        }
        
        # Save both latest and numbered checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
        with open(latest_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        if iteration % self.checkpoint_interval == 0:
            iter_path = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration}.json")
            with open(iter_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        
        # Also save the GP model
        if iteration % self.checkpoint_interval == 0:
            model_path = os.path.join(self.checkpoint_dir, f"gp_model_iter_{iteration}.pkl")
            joblib.dump(self.gp, model_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load optimization state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        self.X_samples = [np.array(x) for x in checkpoint["X_samples"]]
        self.y_samples = checkpoint["y_samples"]
        self.y_raw = checkpoint["y_raw"]
        self.random_state = checkpoint["random_state"]
        
        # If there are enough samples, fit the GP
        if len(self.X_samples) >= 2:
            X = np.vstack(self.X_samples)
            y = np.array(self.y_samples)
            self.gp.fit(X, y)
        
        logger.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        logger.info(f"Best value so far: {checkpoint['best_value']}")
    
    def optimize(self, n_iterations: int, early_stopping: Optional[Dict] = None) -> OptimizationResult:
        """
        Run the Bayesian optimization process.
        
        Args:
            n_iterations: Number of optimization iterations
            early_stopping: Dict with early stopping criteria, e.g., {"patience": 10, "min_improvement": 0.001}
            
        Returns:
            OptimizationResult object with the results
        """
        start_time = time.time()
        
        try:
            # If we don't have any samples yet, evaluate initial points
            if len(self.X_samples) == 0:
                self._evaluate_initial_points()
            
            # If we still don't have enough valid samples, raise an error
            if len(self.X_samples) < 2:
                raise OptimizationError("Not enough valid initial points to build a model")
            
            # Fit the GP model with existing data
            X = np.vstack(self.X_samples)
            y = np.array(self.y_samples)
            self.gp.fit(X, y)
            
            # Track best solution so far
            idx_best = np.argmin(y) if self.optimization_direction == "minimize" else np.argmax(y)
            y_best = y[idx_best]
            best_params = self._array_to_params(X[idx_best])
            best_raw_value = self.y_raw[idx_best]
            
            # Setup early stopping if requested
            no_improvement_count = 0
            if early_stopping:
                patience = early_stopping.get("patience", 10)
                min_improvement = early_stopping.get("min_improvement", 0.001)
            
            # For storing acquisition values
            acquisition_values = []
            
            # Main optimization loop
            for i in range(n_iterations):
                iteration = i + 1
                logger.info(f"Starting iteration {iteration}/{n_iterations}")
                
                # Determine next points to evaluate
                next_points = []
                next_acq_values = []
                
                # Get current best for acquisition function
                if self.acquisition_function in ["EI", "POI"]:
                    # For EI and POI, we need the best observed value
                    acquisition_func = self._get_acquisition_function(y_best=y_best)
                else:
                    # For UCB, we don't need the best observed value
                    acquisition_func = self._get_acquisition_function()
                
                # Get next evaluation points (multiple if parallel)
                for _ in range(self.parallel_evaluations):
                    x_next, acq_value = self._optimize_acquisition(acquisition_func)
                    next_points.append(self._array_to_params(x_next))
                    next_acq_values.append(acq_value)
                    
                    # Add a temporary point to avoid selecting the same point multiple times
                    if _ < self.parallel_evaluations - 1:
                        self.X_samples.append(x_next)
                        self.y_samples.append(y_best)  # Use current best as a placeholder
                        self.gp.fit(np.vstack(self.X_samples), np.array(self.y_samples))
                
                # Remove the temporary points
                if self.parallel_evaluations > 1:
                    self.X_samples = self.X_samples[:-self.parallel_evaluations+1]
                    self.y_samples = self.y_samples[:-self.parallel_evaluations+1]
                
                # Store acquisition values
                acquisition_values.extend(next_acq_values)
                
                # Evaluate the new points
                if self.parallel_evaluations > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_evaluations) as executor:
                        results = list(executor.map(self._evaluate_function, next_points))
                else:
                    results = [self._evaluate_function(p) for p in next_points]
                
                # Process the results
                for j, (signed_result, raw_result) in enumerate(results):
                    if raw_result is not None:  # Only process valid results
                        # Store the result
                        self.X_samples.append(self._params_to_array(next_points[j]))
                        self.y_samples.append(signed_result)
                        self.y_raw.append(raw_result)
                        
                        # Check if this is a new best
                        current_best = y_best
                        if (self.optimization_direction == "minimize" and signed_result < y_best) or \
                           (self.optimization_direction == "maximize" and signed_result > y_best):
                            y_best = signed_result
                            best_params = next_points[j]
                            best_raw_value = raw_result
                            no_improvement_count = 0
                            logger.info(f"New best found at iteration {iteration}: {best_raw_value}")
                        else:
                            no_improvement_count += 1
                
                # Update the GP model with all data
                X = np.vstack(self.X_samples)
                y = np.array(self.y_samples)
                self.gp.fit(X, y)
                
                # Display current status
                if self.verbose:
                    logger.info(f"Iteration {iteration}/{n_iterations} complete")
                    logger.info(f"Best value: {best_raw_value}")
                    logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
                
                # Save checkpoint
                self._save_checkpoint(iteration, best_params, best_raw_value)
                
                # Check for early stopping
                if early_stopping and no_improvement_count >= patience:
                    logger.info(f"Early stopping triggered after {iteration} iterations")
                    break
            
            # Optimization completed successfully
            elapsed_time = time.time() - start_time
            
            result = OptimizationResult(
                best_params=best_params,
                best_value=best_raw_value,
                all_params=[self._array_to_params(x) for x in X],
                all_values=self.y_raw,
                acquisition_values=acquisition_values,
                optimization_time=elapsed_time,
                iterations=n_iterations,
                success=True
            )
            
            logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            logger.info(f"Best value: {best_raw_value}")
            logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Optimization failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to salvage the best result so far if we have any
            best_params = {}
            best_value = float('-inf') if self.optimization_direction == "maximize" else float('inf')
            
            if len(self.X_samples) > 0:
                X = np.vstack(self.X_samples)
                y = np.array(self.y_samples)
                idx_best = np.argmin(y) if self.optimization_direction == "minimize" else np.argmax(y)
                best_params = self._array_to_params(X[idx_best])
                best_value = self.y_raw[idx_best]
            
            return OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                all_params=[self._array_to_params(x) for x in self.X_samples] if len(self.X_samples) > 0 else [],
                all_values=self.y_raw,
                optimization_time=elapsed_time,
                iterations=len(self.X_samples),
                success=False,
                error_message=str(e)
            )
    
    def plot_optimization_results(self, result: OptimizationResult, save_path: Optional[str] = None):
        """
        Plot the optimization results.
        
        Args:
            result: OptimizationResult object
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 1: Objective value vs iteration
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(range(1, len(result.all_values) + 1), result.all_values, 'b-', marker='o')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Objective Value vs Iteration')
            ax1.grid(True)
            
            # Plot 2: Acquisition values vs iteration
            if result.acquisition_values:
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.plot(range(1, len(result.acquisition_values) + 1), result.acquisition_values, 'r-', marker='x')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Acquisition Value')
                ax2.set_title('Acquisition Value vs Iteration')
                ax2.grid(True)
            
            # Plot 3: Parameter values vs iteration for top parameters
            ax3 = fig.add_subplot(2, 2, 3)
            for param_name in self.param_names[:5]:  # Plot up to 5 parameters
                values = [params[param_name] for params in result.all_params]
                ax3.plot(range(1, len(values) + 1), values, marker='.', label=param_name)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Parameter Values vs Iteration')
            ax3.grid(True)
            ax3.legend()
            
            # Plot 4: 3D surface for two most important parameters
            if len(self.param_names) >= 2 and len(result.all_values) >= 10:
                ax4 = fig.add_subplot(2, 2, 4, projection='3d')
                
                # Choose the two parameters with most variance
                param_variances = []
                for param_name in self.param_names:
                    values = [params[param_name] for params in result.all_params]
                    param_variances.append((param_name, np.var(values)))
                
                param_variances.sort(key=lambda x: x[1], reverse=True)
                param1, param2 = param_variances[0][0], param_variances[1][0]
                
                # Extract data for these parameters
                x = [params[param1] for params in result.all_params]
                y = [params[param2] for params in result.all_params]
                z = result.all_values
                
                # Create a meshgrid for surface plotting
                from scipy.interpolate import griddata
                
                x_range = np.linspace(min(x), max(x), 20)
                y_range = np.linspace(min(y), max(y), 20)
                X, Y = np.meshgrid(x_range, y_range)
                Z = griddata((x, y), z, (X, Y), method='cubic')
                
                # Plot the surface and scatter points
                surf = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                ax4.scatter(x, y, z, c='r', marker='o')
                
                ax4.set_xlabel(param1)
                ax4.set_ylabel(param2)
                ax4.set_zlabel('Objective Value')
                ax4.set_title(f'Objective Surface for {param1} and {param2}')
                fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting optimization results: {str(e)}")
            return None


class StrategyOptimizer:
    """
    Specialized Bayesian optimizer for trading strategy optimization.
    Integrates with backtesting to tune strategy parameters for optimal performance.
    """
    
    def __init__(
        self,
        strategy_class,
        param_bounds: Dict[str, Tuple[float, float]],
        backtest_func: Callable,
        optimization_metric: str = "sharpe_ratio",
        optimization_direction: str = "maximize",
        initial_capital: float = 10000.0,
        time_period: str = "1y",
        assets: List[str] = None,
        n_initial_points: int = 5,
        max_iterations: int = 50,
        parallel_evaluations: int = 1,
        gp_kernel: str = DEFAULT_GP_KERNEL,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the Strategy Optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
            param_bounds: Dictionary mapping parameter names to (lower, upper) bounds
            backtest_func: Function to run backtest with strategy parameters
            optimization_metric: Metric to optimize (sharpe_ratio, sortino_ratio, return, etc)
            optimization_direction: Either "maximize" or "minimize"
            initial_capital: Initial capital for backtesting
            time_period: Time period for backtesting
            assets: List of assets to backtest on
            n_initial_points: Number of initial random points to evaluate
            max_iterations: Maximum number of optimization iterations
            parallel_evaluations: Number of parallel evaluations
            gp_kernel: Kernel type for Gaussian Process
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print verbose output
        """
        self.strategy_class = strategy_class
        self.param_bounds = param_bounds
        self.backtest_func = backtest_func
        self.optimization_metric = optimization_metric
        self.optimization_direction = optimization_direction
        self.initial_capital = initial_capital
        self.time_period = time_period
        self.assets = assets or []
        self.n_initial_points = n_initial_points
        self.max_iterations = max_iterations
        self.parallel_evaluations = parallel_evaluations
        self.gp_kernel = gp_kernel
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        # Set up checkpoint directory
        if checkpoint_dir:
            checkpoint_dir = os.path.join(checkpoint_dir, f"{strategy_class.__name__}_opt")
            create_directory(checkpoint_dir)
        
        logger.info(f"Initializing strategy optimizer for {strategy_class.__name__}")
        logger.info(f"Optimization metric: {optimization_metric}")
        logger.info(f"Assets: {', '.join(assets) if assets else 'None specified'}")
    
    def _objective_function(self, **params):
        """
        Objective function for the optimizer. Runs backtest with given parameters.
        
        Args:
            **params: Strategy parameters
            
        Returns:
            Performance metric value
        """
        try:
            # Create strategy instance with these parameters
            strategy = self.strategy_class(**params)
            
            # Run backtest
            backtest_results = self.backtest_func(
                strategy=strategy,
                initial_capital=self.initial_capital,
                time_period=self.time_period,
                assets=self.assets
            )
            
            # Calculate the metric
            if self.optimization_metric == "sharpe_ratio":
                metric = compute_sharpe_ratio(backtest_results)
            elif self.optimization_metric == "sortino_ratio":
                metric = compute_sortino_ratio(backtest_results)
            elif self.optimization_metric == "return":
                metric = backtest_results.get("total_return", 0.0)
            elif self.optimization_metric == "max_drawdown":
                metric = backtest_results.get("max_drawdown", 100.0)
            elif self.optimization_metric == "profit_factor":
                metric = backtest_results.get("profit_factor", 0.0)
            else:
                metric = backtest_results.get(self.optimization_metric, 0.0)
            
            logger.debug(f"Parameters: {params}, Metric: {metric}")
            return metric
            
        except Exception as e:
            logger.error(f"Error in objective function with parameters {params}: {str(e)}")
            return float('-inf') if self.optimization_direction == "maximize" else float('inf')
    
    def optimize(self, n_iterations: Optional[int] = None, early_stopping: Optional[Dict] = None) -> OptimizationResult:
        """
        Run the strategy optimization process.
        
        Args:
            n_iterations: Number of optimization iterations (defaults to self.max_iterations)
            early_stopping: Early stopping criteria
            
        Returns:
            OptimizationResult object with the results
        """
        n_iterations = n_iterations or self.max_iterations
        
        # Create Bayesian optimizer
        optimizer = BayesianOptimizer(
            param_bounds=self.param_bounds,
            objective_function=self._objective_function,
            optimization_direction=self.optimization_direction,
            kernel_type=self.gp_kernel,
            acquisition_function="EI",
            initial_points=self.n_initial_points,
            verbose=self.verbose,
            checkpoint_dir=self.checkpoint_dir,
            parallel_evaluations=self.parallel_evaluations
        )
        
        # Run optimization
        result = optimizer.optimize(n_iterations=n_iterations, early_stopping=early_stopping)
        
        # Save optimization results
        if self.checkpoint_dir:
            result_path = os.path.join(self.checkpoint_dir, "optimization_result.json")
            result.save(result_path)
            
            # Also plot if matplotlib is available
            try:
                plot_path = os.path.join(self.checkpoint_dir, "optimization_plot.png")
                optimizer.plot_optimization_results(result, save_path=plot_path)
            except:
                logger.warning("Could not create optimization plot")
        
        return result
    
    def optimize_multiple_assets(self, asset_weights: Optional[Dict[str, float]] = None) -> Dict[str, OptimizationResult]:
        """
        Optimize strategy parameters individually for multiple assets.
        
        Args:
            asset_weights: Optional dictionary mapping assets to weights for combined optimization
            
        Returns:
            Dictionary mapping assets to their optimization results
        """
        results = {}
        
        # Optimize for each asset individually
        for asset in self.assets:
            logger.info(f"Optimizing strategy for asset: {asset}")
            
            # Create asset-specific optimizer
            asset_optimizer = StrategyOptimizer(
                strategy_class=self.strategy_class,
                param_bounds=self.param_bounds,
                backtest_func=self.backtest_func,
                optimization_metric=self.optimization_metric,
                optimization_direction=self.optimization_direction,
                initial_capital=self.initial_capital,
                time_period=self.time_period,
                assets=[asset],
                n_initial_points=self.n_initial_points,
                max_iterations=self.max_iterations,
                parallel_evaluations=self.parallel_evaluations,
                gp_kernel=self.gp_kernel,
                checkpoint_dir=self.checkpoint_dir and os.path.join(self.checkpoint_dir, asset),
                verbose=self.verbose
            )
            
            # Run optimization
            asset_result = asset_optimizer.optimize()
            results[asset] = asset_result
        
        return results


class MultiObjectiveOptimizer:
    """
    Multi-objective Bayesian optimization for trading strategies.
    Optimizes multiple objectives simultaneously (e.g., returns and drawdown).
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        objective_functions: Dict[str, Callable],
        objective_weights: Dict[str, float],
        objective_directions: Dict[str, str],
        n_initial_points: int = 10,
        exploration_weight: float = 0.1,
        kernel_type: str = DEFAULT_GP_KERNEL,
        checkpoint_dir: Optional[str] = None,
        parallel_evaluations: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the Multi-Objective Optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (lower, upper) bounds
            objective_functions: Dictionary mapping objective names to functions
            objective_weights: Dictionary mapping objective names to weights
            objective_directions: Dictionary mapping objective names to "maximize" or "minimize"
            n_initial_points: Number of initial random points to evaluate
            exploration_weight: Weight for exploration vs exploitation
            kernel_type: Kernel type for Gaussian Process
            checkpoint_dir: Directory to save checkpoints
            parallel_evaluations: Number of parallel evaluations
            verbose: Whether to print verbose output
        """
        self.param_bounds = param_bounds
        self.objective_functions = objective_functions
        self.objective_weights = objective_weights
        self.objective_directions = objective_directions
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        self.kernel_type = kernel_type
        self.checkpoint_dir = checkpoint_dir
        self.parallel_evaluations = parallel_evaluations
        self.verbose = verbose
        
        # Validate inputs
        for obj_name in objective_functions:
            if obj_name not in objective_weights:
                raise ValueError(f"Missing weight for objective {obj_name}")
            if obj_name not in objective_directions:
                raise ValueError(f"Missing direction for objective {obj_name}")
            if objective_directions[obj_name] not in OPTIMIZATION_DIRECTION:
                raise ValueError(f"Invalid direction for objective {obj_name}")
        
        # Set up checkpoint directory
        if checkpoint_dir:
            create_directory(checkpoint_dir)
        
        # Create individual optimizers for each objective
        self.optimizers = {}
        for obj_name, obj_func in objective_functions.items():
            self.optimizers[obj_name] = BayesianOptimizer(
                param_bounds=param_bounds,
                objective_function=obj_func,
                optimization_direction=objective_directions[obj_name],
                kernel_type=kernel_type,
                acquisition_function="EI",
                initial_points=n_initial_points,
                exploration_weight=exploration_weight,
                verbose=verbose,
                checkpoint_dir=checkpoint_dir and os.path.join(checkpoint_dir, obj_name),
                parallel_evaluations=parallel_evaluations
            )
        
        logger.info(f"Initialized Multi-Objective Optimizer with {len(objective_functions)} objectives")
        
    def _combined_objective(self, **params):
        """
        Combined weighted objective function.
        
        Args:
            **params: Parameters to evaluate
            
        Returns:
            Weighted sum of normalized objective values
        """
        results = {}
        combined_value = 0.0
        
        # Evaluate each objective
        for obj_name, obj_func in self.objective_functions.items():
            try:
                value = obj_func(**params)
                
                # Store raw value
                results[obj_name] = value
                
                # Normalize based on direction (we want to maximize the combined objective)
                sign = 1 if self.objective_directions[obj_name] == "maximize" else -1
                weighted_value = sign * value * self.objective_weights[obj_name]
                
                combined_value += weighted_value
                
            except Exception as e:
                logger.error(f"Error evaluating {obj_name} with params {params}: {str(e)}")
                return float('-inf')
        
        return combined_value, results
    
    def optimize(self, n_iterations: int = 50, early_stopping: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the multi-objective optimization process.
        
        Args:
            n_iterations: Number of optimization iterations
            early_stopping: Early stopping criteria
            
        Returns:
            Dictionary with optimization results
        """
        # Create optimizer for combined objective
        combined_optimizer = BayesianOptimizer(
            param_bounds=self.param_bounds,
            objective_function=lambda **params: self._combined_objective(**params)[0],
            optimization_direction="maximize",
            kernel_type=self.kernel_type,
            acquisition_function="EI",
            initial_points=self.n_initial_points,
            verbose=self.verbose,
            checkpoint_dir=self.checkpoint_dir and os.path.join(self.checkpoint_dir, "combined"),
            parallel_evaluations=self.parallel_evaluations
        )
        
        # Run optimization
        result = combined_optimizer.optimize(n_iterations=n_iterations, early_stopping=early_stopping)
        
        # Get individual objective values for the best parameters
        best_params = result.best_params
        _, objective_values = self._combined_objective(**best_params)
        
        # Prepare full result
        full_result = {
            "best_params": best_params,
            "combined_value": result.best_value,
            "objective_values": objective_values,
            "optimization_time": result.optimization_time,
            "iterations": result.iterations,
            "success": result.success
        }
        
        # Save results
        if self.checkpoint_dir:
            result_path = os.path.join(self.checkpoint_dir, "multi_objective_result.json")
            with open(result_path, 'w') as f:
                json.dump(full_result, f, indent=2)
        
        return full_result
        
    def optimize_pareto(self, n_iterations: int = 50) -> Dict[str, Any]:
        """
        Find the Pareto frontier through multiple optimizations.
        
        Args:
            n_iterations: Number of optimization iterations per objective
            
        Returns:
            Dictionary with Pareto frontier results
        """
        # First, optimize each objective individually
        individual_results = {}
        for obj_name, optimizer in self.optimizers.items():
            logger.info(f"Optimizing objective: {obj_name}")
            result = optimizer.optimize(n_iterations=n_iterations)
            individual_results[obj_name] = result
        
        # Then optimize combined objective
        logger.info("Optimizing combined objective")
        combined_result = self.optimize(n_iterations=n_iterations)
        
        # Combine all parameter sets and evaluate all objectives
        all_params = []
        all_objective_values = []
        
        # Add individual optima
        for obj_name, result in individual_results.items():
            all_params.append(result.best_params)
        
        # Add combined optimum
        all_params.append(combined_result["best_params"])
        
        # Add some random parameter combinations for diversity
        for _ in range(10):
            random_params = {}
            for param_name, (lower, upper) in self.param_bounds.items():
                random_params[param_name] = lower + (upper - lower) * np.random.random()
            all_params.append(random_params)
        
        # Evaluate all objectives for all parameter sets
        for params in all_params:
            obj_values = {}
            for obj_name, obj_func in self.objective_functions.items():
                try:
                    value = obj_func(**params)
                    obj_values[obj_name] = value
                except:
                    obj_values[obj_name] = float('-inf') if self.objective_directions[obj_name] == "maximize" else float('inf')
            
            all_objective_values.append(obj_values)
        
        # Find Pareto-optimal points
        pareto_optimal = []
        pareto_params = []
        
        for i, (params, obj_values) in enumerate(zip(all_params, all_objective_values)):
            is_dominated = False
            
            for j, other_obj_values in enumerate(all_objective_values):
                if i == j:
                    continue
                
                dominates = True
                for obj_name in self.objective_functions:
                    # Check if j dominates i for this objective
                    if self.objective_directions[obj_name] == "maximize":
                        if other_obj_values[obj_name] < obj_values[obj_name]:
                            dominates = False
                            break
                    else:  # minimize
                        if other_obj_values[obj_name] > obj_values[obj_name]:
                            dominates = False
                            break
                
                # If j dominates i, mark i as dominated and break
                if dominates:
                    is_dominated = True
                    break
            
            # If i is not dominated by any other point, add to Pareto set
            if not is_dominated:
                pareto_optimal.append(obj_values)
                pareto_params.append(params)
        
        # Prepare result
        pareto_result = {
            "individual_results": {name: result.to_dict() for name, result in individual_results.items()},
            "combined_result": combined_result,
            "pareto_frontier": [{"params": params, "values": values} 
                               for params, values in zip(pareto_params, pareto_optimal)]
        }
        
        # Save results
        if self.checkpoint_dir:
            result_path = os.path.join(self.checkpoint_dir, "pareto_result.json")
            with open(result_path, 'w') as f:
                json.dump(pareto_result, f, indent=2)
        
        return pareto_result
