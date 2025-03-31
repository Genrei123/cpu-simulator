#!/usr/bin/env python3
"""
CPU SIMULATION PROGRAM - DOCUMENTATION

OVERVIEW:
This program simulates a simplified CPU executing algorithms while tracking register values, 
cache behavior, and memory usage. It visualizes these components to help students understand 
basic computer architecture concepts.

COMPONENTS:
1. CPU Simulation:
   - Registers (R1, R2, R3, ACC): Store intermediate values during execution
   - Cache: Simulates fast memory with limited capacity (8 entries)
   - Memory: Divided into stack (for local variables) and heap (for dynamic allocations)

2. Visualization:
   - Register values over time
   - Cache hit/miss performance
   - Memory usage (stack vs heap)

EXECUTION FLOW:
1. Loads an algorithm from a Python file
2. Simulates execution while tracking CPU state
3. Generates interactive visualizations with explanations
4. Displays results with educational annotations

REALISM AND LIMITATIONS:
- This is a simplified educational model, not a cycle-accurate CPU emulation
- Register behavior is abstracted (no real instruction set)
- Cache uses simple LRU-like replacement
- Memory model separates stack/heap but doesn't simulate addresses precisely
- Timing is abstract (no real clock cycles)
- Visualization shows conceptual behavior rather than precise hardware details

USAGE:
python main.py [algorithm_file.py] [arguments]
Example: python main.py factorial.py 5
"""

import sys
import os
import importlib.util
import time
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Tuple

class Register:
    """Simulates a CPU register with history tracking"""
    def __init__(self, name: str):
        self.name = name
        self.value = 0
        self.history = []
    
    def set(self, value: int) -> None:
        """Set register value and record history"""
        self.value = value
        self.history.append(value)
    
    def get(self) -> int:
        """Get current register value"""
        return self.value

class Cache:
    """Simulates a simple CPU cache with hit/miss tracking"""
    def __init__(self, size: int = 8):
        self.size = size
        self.entries = {}  # key: memory address, value: data
        self.hits = []
        self.misses = []
        self.hit_count = 0
        self.miss_count = 0
    
    def access(self, address: str, write_value: Any = None) -> Any:
        """Access cache, return value and record hit/miss"""
        if address in self.entries:
            # Cache hit
            self.hit_count += 1
            self.hits.append(self.hit_count + self.miss_count)
            if write_value is not None:
                self.entries[address] = write_value
            return self.entries[address]
        else:
            # Cache miss
            self.miss_count += 1
            self.misses.append(self.hit_count + self.miss_count)
            if write_value is not None:
                # If cache is full, remove random entry (simplified LRU)
                if len(self.entries) >= self.size:
                    oldest = next(iter(self.entries))
                    del self.entries[oldest]
                self.entries[address] = write_value
            return None

class Memory:
    """Simulates RAM with stack and heap sections"""
    def __init__(self):
        self.stack = {}  # For local variables
        self.heap = {}   # For dynamic allocations
        self.stack_usage = []
        self.heap_usage = []
    
    def stack_allocate(self, var_name: str, value: Any) -> None:
        """Allocate variable on stack"""
        self.stack[var_name] = value
        self.stack_usage.append(len(self.stack))
    
    def heap_allocate(self, address: str, value: Any) -> None:
        """Allocate memory on heap"""
        self.heap[address] = value
        self.heap_usage.append(len(self.heap))
    
    def stack_access(self, var_name: str) -> Any:
        """Access stack variable"""
        return self.stack.get(var_name)
    
    def heap_access(self, address: str) -> Any:
        """Access heap memory"""
        return self.heap.get(address)

class CPU:
    """Main CPU simulator class"""
    def __init__(self):
        # Initialize CPU components
        self.registers = {
            "R1": Register("R1"),
            "R2": Register("R2"),
            "R3": Register("R3"),
            "ACC": Register("ACC"),  # Accumulator
        }
        self.cache = Cache()
        self.memory = Memory()
        self.execution_steps = []
        self.step_count = 0
    
    def execute_algorithm(self, algorithm_module, *args) -> Any:
        """Execute the provided algorithm with CPU instrumentation"""
        algorithm_name = next((name for name in dir(algorithm_module) 
                             if callable(getattr(algorithm_module, name)) 
                             and not name.startswith('_')), None)
        
        if not algorithm_name:
            raise ValueError("No algorithm function found in the provided module")
        
        algorithm_func = getattr(algorithm_module, algorithm_name)
        
        # Start instrumented execution
        self._start_execution()
        result = self._execute_with_instrumentation(algorithm_func, *args)
        return result
    
    def _start_execution(self) -> None:
        """Reset CPU state before execution"""
        for reg in self.registers.values():
            reg.set(0)
        self.step_count = 0
    
    def _execute_with_instrumentation(self, func, *args) -> Any:
        """Execute the algorithm with CPU instrumentation"""
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Store function arguments in registers and stack
        for i, (arg_name, arg_value) in enumerate(zip(arg_names, args)):
            reg_name = f"R{i+1}" if f"R{i+1}" in self.registers else "ACC"
            self.registers[reg_name].set(arg_value)
            self.memory.stack_allocate(arg_name, arg_value)
            self._record_step(f"Initialize {reg_name} with argument {arg_name}={arg_value}")
        
        # Special handling for factorial to demonstrate register usage
        if func.__name__ == "factorial":
            n = args[0]
            result = 1
            self.registers["ACC"].set(result)
            
            for i in range(1, n + 1):
                self._record_step(f"Calculate factorial step: {i}")
                self.registers["R1"].set(i)
                self._record_step(f"Load {i} into R1")
                
                current_result = self.registers["ACC"].get()
                self._record_step(f"Access current result {current_result} from ACC")
                
                cache_address = f"temp_{i}"
                self.cache.access(cache_address, i)
                
                new_result = current_result * i
                self._record_step(f"Multiply ACC({current_result}) * R1({i}) = {new_result}")
                
                self.registers["ACC"].set(new_result)
                self._record_step(f"Store result {new_result} in ACC")
                
                self.memory.heap_allocate(f"result_{i}", new_result)
                time.sleep(0.1)
            
            result = self.registers["ACC"].get()
        else:
            # Generic execution for other algorithms
            result = func(*args)
            self.registers["ACC"].set(result)
            self._record_step(f"Store final result {result} in ACC")
        
        return result
    
    def _record_step(self, description: str) -> None:
        """Record a CPU execution step"""
        self.step_count += 1
        self.execution_steps.append({
            "step": self.step_count,
            "description": description,
            "registers": {name: reg.get() for name, reg in self.registers.items()},
            "cache_hits": self.cache.hit_count,
            "cache_misses": self.cache.miss_count,
            "stack_size": len(self.memory.stack),
            "heap_size": len(self.memory.heap)
        })

class CPUVisualizer:
    """Handles visualization of CPU simulation results with embedded explanations"""
    def __init__(self, cpu: CPU):
        self.cpu = cpu
        self.fig = plt.figure(figsize=(15, 12))
        self.fig.suptitle("CPU Simulation Visualizations", fontsize=16, y=1.02)
        self.gs = GridSpec(3, 1, figure=self.fig)
    
    def generate_visualizations(self) -> None:
        """Generate and display all visualizations with explanations"""
        self._create_register_plot()
        self._create_cache_plot()
        self._create_memory_plot()
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
    
    def _create_register_plot(self) -> None:
        """Create register values plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[0])
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        
        # Plot register values
        for reg_name, register in self.cpu.registers.items():
            if len(register.history) < len(steps):
                register.history.extend([register.history[-1]] * (len(steps) - len(register.history)))
            ax.plot(steps, register.history[:len(steps)], marker='o', label=reg_name)
        
        # Add educational annotations
        ax.set_title("Register Values During Execution", pad=20)
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Register Value")
        ax.legend()
        ax.grid(True)
        
        # Add explanatory text
        explanation = (
            "Registers are the CPU's fastest storage locations. In this simulation:\n"
            "- R1-R3: General purpose registers for intermediate values\n"
            "- ACC: Accumulator stores the main computation result\n"
            "Note how values change during algorithm execution."
        )
        ax.text(0.02, -0.25, explanation, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    def _create_cache_plot(self) -> None:
        """Create cache performance plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[1])
        
        # Extract cache data
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        hits = [step["cache_hits"] for step in self.cpu.execution_steps]
        misses = [step["cache_misses"] for step in self.cpu.execution_steps]
        
        # Calculate hit rate
        hit_rates = []
        for i, (hit, miss) in enumerate(zip(hits, misses)):
            if hit + miss == 0:
                hit_rates.append(0)
            else:
                hit_rates.append(hit / (hit + miss) * 100)
        
        # Plot hit rate
        ax.plot(steps, hit_rates, marker='o', color='green', label='Cache Hit Rate (%)')
        
        # Add cumulative hits and misses
        ax.bar([1], [hits[-1]], width=0.4, alpha=0.6, color='blue', label=f'Cache Hits ({hits[-1]})')
        ax.bar([2], [misses[-1]], width=0.4, alpha=0.6, color='red', label=f'Cache Misses ({misses[-1]})')
        
        # Add educational annotations
        ax.set_title("Cache Performance", pad=20)
        ax.set_xlabel("Execution")
        ax.set_ylabel("Hit Rate (%)")
        ax.legend()
        ax.grid(True)
        
        # Add explanatory text
        explanation = (
            "Cache is fast memory that stores recently used data:\n"
            "- Hits (blue): Data found in cache (fast access)\n"
            "- Misses (red): Data not in cache (slower memory access required)\n"
            "Good algorithms maximize cache hits for better performance."
        )
        ax.text(0.02, -0.25, explanation, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    def _create_memory_plot(self) -> None:
        """Create memory usage plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[2])
        
        # Extract memory data
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        stack_usage = [step["stack_size"] for step in self.cpu.execution_steps]
        heap_usage = [step["heap_size"] for step in self.cpu.execution_steps]
        
        # Create stacked area chart
        ax.fill_between(steps, 0, stack_usage, alpha=0.5, color='blue', label='Stack Usage')
        ax.fill_between(steps, stack_usage, [s + h for s, h in zip(stack_usage, heap_usage)], 
                       alpha=0.5, color='green', label='Heap Usage')
        
        # Add educational annotations
        ax.set_title("Memory Usage During Execution", pad=20)
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Number of Allocations")
        ax.legend()
        ax.grid(True)
        
        # Add explanatory text
        explanation = (
            "Memory is divided into regions:\n"
            "- Stack (blue): Stores local variables and function calls\n"
            "- Heap (green): Stores dynamically allocated data\n"
            "Note how different algorithms use memory differently."
        )
        ax.text(0.02, -0.25, explanation, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

class AIAssistant:
    """Provides educational descriptions of simulation results"""
    def __init__(self):
        self.api_key = None  # TODO: Insert your ChatGPT 4o API key here
    
    def describe_visualization(self, plot_type: str) -> str:
        """Provide educational description for each plot type"""
        descriptions = {
            'registers': (
                "REGISTER ANALYSIS:\n"
                "The register plot shows how values change in the CPU's fastest storage locations. "
                "The accumulator (ACC) typically holds the main computation result, while R1-R3 "
                "store intermediate values. In recursive algorithms, you'll see more register "
                "activity as values are constantly updated."
            ),
            'cache': (
                "CACHE ANALYSIS:\n"
                "This plot shows memory access efficiency. Cache hits (blue) occur when the CPU "
                "finds data in its fast cache memory. Misses (red) require slower main memory access. "
                "Algorithms with good locality of reference (accessing nearby memory locations) "
                "will show higher hit rates."
            ),
            'memory': (
                "MEMORY ANALYSIS:\n"
                "Memory usage is divided between stack (blue) and heap (green). The stack grows "
                "with each function call and stores local variables. The heap stores dynamically "
                "allocated data. Recursive algorithms typically show more stack usage, while "
                "algorithms that build data structures show more heap usage."
            )
        }
        return descriptions.get(plot_type, "No description available for this plot type.")

def load_algorithm(file_path: str):
    """Load algorithm module from file path"""
    if not os.path.exists(file_path):
        print(f"Error: Algorithm file '{file_path}' not found")
        sys.exit(1)
    
    try:
        spec = importlib.util.spec_from_file_location("algorithm", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading algorithm: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the CPU simulation"""
    if len(sys.argv) < 2:
        print("Usage: python main.py [algorithm_file] [algorithm_args]")
        print("Example: python main.py factorial.py 5")
        sys.exit(1)
    
    algorithm_file = sys.argv[1]
    algorithm_args = []
    
    # Parse additional arguments
    for arg in sys.argv[2:]:
        try:
            algorithm_args.append(int(arg))
        except ValueError:
            algorithm_args.append(arg)
    
    # Load the algorithm
    algorithm_module = load_algorithm(algorithm_file)
    
    # Initialize CPU and execute algorithm
    cpu = CPU()
    
    try:
        print(f"\nExecuting algorithm from {algorithm_file} with args {algorithm_args}...")
        result = cpu.execute_algorithm(algorithm_module, *algorithm_args)
        print(f"Algorithm execution completed. Result: {result}\n")
        
        # Generate and display visualizations
        print("Displaying CPU simulation visualizations...")
        visualizer = CPUVisualizer(cpu)
        visualizer.generate_visualizations()
        
        # Provide AI descriptions
        ai_assistant = AIAssistant()
        print("\n=== EDUCATIONAL ANALYSIS ===")
        print(ai_assistant.describe_visualization('registers'))
        print(ai_assistant.describe_visualization('cache'))
        print(ai_assistant.describe_visualization('memory'))
    
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()