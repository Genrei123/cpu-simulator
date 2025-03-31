#!/usr/bin/env python3
"""
CPU SIMULATION PROGRAM - DOCUMENTATION V2.0

OVERVIEW:
This program simulates a simplified CPU executing arbitrary Python algorithms while
tracking simulated register values, cache behavior, and memory usage. It uses Python's
`sys.settrace` mechanism to intercept execution events (function calls, line execution,
returns) and simulates corresponding CPU-level actions. The goal is to provide an
educational tool for students learning basic computer architecture concepts by
visualizing these abstract behaviors.

COMPONENTS:
1. CPU Simulation (Abstract Machine):
   - Registers (R1, R2, R3, ACC): Abstracted storage for intermediate values.
     Their usage is inferred from execution events (e.g., ACC for results, R1/R2 for operands).
   - Cache: Simulates a simple cache (8 entries, LRU-like replacement) to model
     memory access speed differences. Cache checks are simulated on inferred memory accesses.
   - Memory: Abstractly divided into stack (tracking function call depth) and heap
     (tracking simulated dynamic allocations, e.g., on variable changes).

2. Visualization:
   - Register values over simulated steps.
   - Cache hit/miss performance over simulated steps.
   - Memory usage (stack depth vs. simulated heap allocations) over simulated steps.

EXECUTION FLOW:
1. Loads a Python algorithm function from a specified file.
2. Sets up a tracing function using `sys.settrace`.
3. Executes the algorithm. The tracer intercepts events (`call`, `line`, `return`).
4. The tracer simulates CPU actions based on these events:
   - `call`: Increments stack depth, simulates argument loading.
   - `line`: Simulates generic computation/memory access (register activity, cache check).
   - `return`: Decrements stack depth, simulates storing result in ACC.
5. Records the state of registers, cache, and memory at each simulated step.
6. Generates interactive visualizations (using Matplotlib) with embedded explanations.
7. Displays results and educational analysis.

REALISM AND LIMITATIONS:
- EDUCATIONAL MODEL: This is a highly simplified, abstract model for teaching, NOT a
  cycle-accurate CPU emulator or a precise performance analysis tool.
- ABSTRACT MACHINE: It doesn't simulate a real instruction set (like x86 or ARM).
  CPU actions are *inferred* conceptually from Python execution events.
- REGISTER ABSTRACTION: Register allocation is simplistic (e.g., cycling R1-R3 for
  general use, ACC for results). It doesn't reflect complex compiler optimizations.
- CACHE SIMULATION: Uses a basic pseudo-LRU cache model. Real cache behavior is far
  more complex (associativity, write policies, coherence). Addresses are variable names
  or simple generated IDs.
- MEMORY MODEL: Stack usage tracks call depth. Heap usage is crudely simulated (e.g.,
  incrementing on variable changes). It doesn't map to real memory addresses or manage
  complex data structures accurately.
- TIMING: Execution steps are based on trace events, not actual clock cycles or
  instruction timings. `time.sleep` has been removed.
- VISUALIZATION: Shows conceptual trends, not precise hardware states.

USAGE:
python main.py [algorithm_file.py] [arguments...]
Example: python main.py factorial.py 5
Example: python main.py sum_list.py 1 2 3 4 5

Factorial Example (`factorial.py`):
def factorial(n):
    if n <= 1:
        return 1
    else:
        # Each operation below triggers simulated steps
        intermediate = factorial(n-1)
        result = n * intermediate
        return result

Sum List Example (`sum_list.py`):
def sum_list(*args):
    total = 0
    # The loop triggers multiple 'line' events
    for x in args:
        total += x
    return total
"""

import sys
import os
import importlib.util
import inspect # Used for tracing
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Tuple, Optional, Callable

# --- Simulation Components ---

class Register:
    """Simulates a CPU register with history tracking"""
    def __init__(self, name: str):
        self.name = name
        self.value: Any = 0 # Allow any type for flexibility
        self.history: List[Any] = [0] # Start with initial value

    def set(self, value: Any) -> None:
        """Set register value and record history"""
        self.value = value
        self.history.append(value)

    def get(self) -> Any:
        """Get current register value"""
        return self.value

class Cache:
    """Simulates a simple CPU cache (pseudo-LRU) with hit/miss tracking"""
    def __init__(self, size: int = 8):
        self.size = size
        # Use list to maintain rough order for pseudo-LRU
        self.entries: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.hit_count = 0
        self.miss_count = 0
        # Store history for plotting
        self.hits_history: List[int] = [0]
        self.misses_history: List[int] = [0]

    def access(self, address: str, write_value: Any = None) -> Tuple[Any, bool]:
        """
        Access cache. Simulate read or write.
        Returns (value, hit_status). Hit_status is True if hit, False if miss.
        """
        hit = False
        value = None

        if address in self.entries:
            # Cache Hit
            self.hit_count += 1
            hit = True
            # Move accessed item to end of list (most recently used)
            self.access_order.remove(address)
            self.access_order.append(address)
            value = self.entries[address]
            if write_value is not None:
                self.entries[address] = write_value # Update value on write hit
                value = write_value
        else:
            # Cache Miss
            self.miss_count += 1
            hit = False
            value = None # On read miss, data isn't in cache yet
            if write_value is not None:
                # Write miss: need to bring data into cache
                if len(self.entries) >= self.size:
                    # Evict least recently used (first element in list)
                    lru_address = self.access_order.pop(0)
                    del self.entries[lru_address]
                # Add new entry
                self.entries[address] = write_value
                self.access_order.append(address)
                value = write_value # Value is now 'in cache' after write

        # Record history for plotting at each access attempt
        self.hits_history.append(self.hit_count)
        self.misses_history.append(self.miss_count)
        return value, hit

    def get_stats(self) -> Tuple[int, int]:
        """Return current hit and miss counts"""
        return self.hit_count, self.miss_count

class Memory:
    """Simulates RAM with abstract stack depth and heap usage tracking"""
    def __init__(self):
        # Track stack depth via function calls
        self.stack_depth = 0
        # Simulate heap allocations abstractly
        self.heap_allocations = 0
        # History for plotting
        self.stack_usage_history: List[int] = [0]
        self.heap_usage_history: List[int] = [0]

    def push_stack_frame(self) -> None:
        """Simulate pushing a stack frame on function call"""
        self.stack_depth += 1
        self.stack_usage_history.append(self.stack_depth)
        self.heap_usage_history.append(self.heap_allocations) # Keep heap history aligned

    def pop_stack_frame(self) -> None:
        """Simulate popping a stack frame on function return"""
        self.stack_depth = max(0, self.stack_depth - 1)
        self.stack_usage_history.append(self.stack_depth)
        self.heap_usage_history.append(self.heap_allocations) # Keep heap history aligned

    def simulate_heap_allocation(self) -> None:
        """Simulate a generic heap allocation"""
        self.heap_allocations += 1
        # Only record history when stack changes or explicitly told to
        # Heap changes are often tied to stack activity in this model

    def record_state(self):
       """Explicitly record current memory state"""
       self.stack_usage_history.append(self.stack_depth)
       self.heap_usage_history.append(self.heap_allocations)


class CPU:
    """
    Main CPU simulator class using sys.settrace for generic algorithm execution.
    Simulates operations based on trace events.
    """
    def __init__(self):
        self.registers = {
            "R1": Register("R1"),
            "R2": Register("R2"),
            "R3": Register("R3"),
            "ACC": Register("ACC"),  # Accumulator
        }
        self.cache = Cache()
        self.memory = Memory()
        self.execution_steps: List[Dict[str, Any]] = []
        self.step_count = 0
        self._current_trace_func: Optional[Callable] = None
        self._original_trace_func: Optional[Callable] = None
        self._temp_reg_counter = 0 # For simple register allocation

    def _get_next_temp_reg(self) -> str:
        """Cycles through R1, R2, R3 for temporary use."""
        reg_names = ["R1", "R2", "R3"]
        reg = reg_names[self._temp_reg_counter % len(reg_names)]
        self._temp_reg_counter += 1
        return reg

    def _record_step(self, description: str) -> None:
        """Record a CPU execution step with current state"""
        self.step_count += 1
        current_registers = {name: reg.get() for name, reg in self.registers.items()}
        cache_hits, cache_misses = self.cache.get_stats()

        self.execution_steps.append({
            "step": self.step_count,
            "description": description,
            "registers": current_registers,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "stack_depth": self.memory.stack_depth,
            "heap_allocations": self.memory.heap_allocations
        })
        # Also ensure register histories are updated even if unchanged
        for reg in self.registers.values():
            if len(reg.history) <= self.step_count:
                 reg.history.append(reg.get()) # Maintain history length

        # Ensure memory history matches step count
        self.memory.record_state()


    def _trace_dispatch(self, frame, event, arg):
        """The tracing function called by sys.settrace."""
        # Avoid tracing into the simulation code itself or libraries
        if frame.f_code.co_filename != self.algorithm_filename:
             return self._trace_dispatch # Continue tracing, but ignore event here

        func_name = frame.f_code.co_name
        line_no = frame.f_lineno

        if event == 'call':
            self.memory.push_stack_frame()
            args = inspect.getargvalues(frame)
            arg_desc = ", ".join(f"{k}={v}" for k, v in args.locals.items() if k in args.args)
            self._record_step(f"CALL: Enter '{func_name}'({arg_desc}) at line {line_no}. Stack depth: {self.memory.stack_depth}")
            # Simulate loading args (simplified)
            # for i, arg_name in enumerate(args.args):
            #     reg_name = f"R{i+1}" if f"R{i+1}" in self.registers else "ACC"
            #     self.registers[reg_name].set(args.locals[arg_name])
            #     self._record_step(f"  Simulate load arg '{arg_name}' ({args.locals[arg_name]}) into {reg_name}")

        elif event == 'line':
            # Simulate generic activity for executing a line
            # This is highly abstract - not decoding Python bytecode
            self._record_step(f"LINE: Execute line {line_no} in '{func_name}'")

            # Simulate a memory read (e.g., access a local variable)
            local_vars = list(frame.f_locals.keys())
            if local_vars:
                var_to_access = random.choice(local_vars)
                # Use variable name as pseudo-address
                _, cache_hit = self.cache.access(f"{func_name}_{var_to_access}")
                hit_miss = "HIT" if cache_hit else "MISS"
                temp_reg = self._get_next_temp_reg()
                # Simulate loading value into a register
                val = frame.f_locals.get(var_to_access, None)
                if isinstance(val, (int, float, str, bool)): # Avoid complex objects in registers
                    self.registers[temp_reg].set(val)
                self._record_step(f"  Simulate READ '{var_to_access}' (Cache {hit_miss}). Load into {temp_reg}")

            # Simulate a computation (results often go to ACC)
            # Just put a placeholder value
            computed_value = random.randint(0, 100) # Dummy computed value
            self.registers["ACC"].set(computed_value)
            self._record_step(f"  Simulate computation. Store result ({computed_value}) in ACC")

            # Simulate a memory write (e.g., update a local variable) - very abstract
            if local_vars:
                var_to_write = random.choice(local_vars)
                val_to_write = self.registers["ACC"].get() # Use ACC value
                # Simulate write to memory (potentially heap for larger changes)
                self.memory.simulate_heap_allocation()
                # Simulate write potentially going through cache
                _, cache_hit = self.cache.access(f"{func_name}_{var_to_write}", write_value=val_to_write)
                hit_miss = "HIT" if cache_hit else "MISS"
                self._record_step(f"  Simulate WRITE '{var_to_write}'={val_to_write} (Cache {hit_miss}, Heap+{1})")


        elif event == 'return':
            self.memory.pop_stack_frame()
            # Simulate placing return value in ACC
            if isinstance(arg, (int, float, str, bool)): # Avoid complex objects
               self.registers["ACC"].set(arg)
            else:
               self.registers["ACC"].set(str(type(arg))) # Represent object type
            self._record_step(f"RETURN: Exit '{func_name}' with value '{arg}'. Stack depth: {self.memory.stack_depth}. Store result in ACC")

        return self._trace_dispatch # Must return the trace function to continue tracing

    def execute_algorithm(self, algorithm_module, algorithm_filename: str, *args) -> Any:
        """Execute the provided algorithm with tracing enabled"""
        self.algorithm_filename = algorithm_filename # Store for tracer filter
        algorithm_name = next((name for name in dir(algorithm_module)
                               if callable(getattr(algorithm_module, name))
                               and not name.startswith('_')
                               # Heuristic: find function defined in the loaded file
                               and getattr(getattr(algorithm_module, name), '__module__', None) == 'algorithm'
                              ), None)

        if not algorithm_name:
             # Fallback if module name check fails (less reliable)
             algorithm_name = next((name for name in dir(algorithm_module)
                                    if callable(getattr(algorithm_module, name))
                                    and not name.startswith('_')), None)

        if not algorithm_name:
            raise ValueError("No callable algorithm function found in the provided module")

        algorithm_func = getattr(algorithm_module, algorithm_name)

        # --- Setup Tracing ---
        self._start_execution()
        self._current_trace_func = self._trace_dispatch
        self._original_trace_func = sys.gettrace() # Save current tracer
        sys.settrace(self._current_trace_func)

        result = None
        try:
            # --- Execute the User's Algorithm ---
            # The tracer (_trace_dispatch) will intercept events during this call
            result = algorithm_func(*args)
            # --- Execution Finished ---
        except Exception as e:
            print(f"\nError during traced execution: {e}")
            raise # Re-raise the exception
        finally:
            # --- Stop Tracing ---
            sys.settrace(self._original_trace_func) # Restore original tracer
            self._current_trace_func = None
            self._record_step(f"END: Algorithm '{algorithm_name}' finished.")

        return result

    def _start_execution(self) -> None:
        """Reset CPU state before execution"""
        self.step_count = 0
        self.execution_steps = []
        self.registers = { name: Register(name) for name in self.registers } # Reset registers
        self.cache = Cache() # Reset cache
        self.memory = Memory() # Reset memory
        self._temp_reg_counter = 0
        # Add initial state step
        self._record_step("START: Initial CPU State")


# --- Visualization ---

class CPUVisualizer:
    """Handles visualization of CPU simulation results with embedded explanations"""
    def __init__(self, cpu: CPU):
        self.cpu = cpu
        self.fig = plt.figure(figsize=(15, 14)) # Increased height for text
        self.fig.suptitle("CPU Simulation Visualization (Abstract Model)", fontsize=16, y=0.99)
        # Adjusted GridSpec for better spacing with text
        self.gs = GridSpec(3, 1, figure=self.fig, hspace=0.6, bottom=0.15)

    def generate_visualizations(self) -> None:
        """Generate and display all visualizations with explanations"""
        if not self.cpu.execution_steps:
            print("No execution steps recorded, cannot generate visualization.")
            return

        self._create_register_plot()
        self._create_cache_plot()
        self._create_memory_plot()

        # Adjust layout and display
        # plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Adjust rect to prevent title overlap and fit text
        plt.show() # Use show() instead of savefig()

    def _get_plot_steps(self) -> List[int]:
         # Ensure we have steps, starting from 0 (initial state) maybe?
         # Let's stick to recorded steps starting from 1
         return list(range(1, len(self.cpu.execution_steps) + 1))


    def _create_register_plot(self) -> None:
        """Create register values plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[0])
        steps = self._get_plot_steps()
        num_steps = len(steps)

        # Plot register values - handle potential missing history entries
        for reg_name, register in self.cpu.registers.items():
             # Pad history if shorter than steps (can happen if register wasn't touched recently)
             history = register.history
             if len(history) < num_steps:
                 padding = [history[-1]] * (num_steps - len(history))
                 history = history + padding
             elif len(history) > num_steps:
                 history = history[:num_steps] # Truncate if too long

             # Only plot numeric types directly
             plot_values = []
             non_numeric_indices = []
             for i, val in enumerate(history):
                 if isinstance(val, (int, float)):
                     plot_values.append(val)
                 else:
                     plot_values.append(None) # Use None for gaps
                     non_numeric_indices.append((i, val))

             ax.plot(steps, plot_values, marker='.', linestyle='-', label=reg_name)

             # Annotate non-numeric values
             # for idx, val in non_numeric_indices:
             #    ax.text(steps[idx], 0, f"{reg_name}={val}", rotation=90, size='x-small', ha='center', va='bottom')


        ax.set_title("Simulated Register Activity", pad=15)
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Register Value (Numeric)")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle=':')

        # Add embedded explanatory text
        explanation = (
            "Registers: CPU's fastest memory. This simulation *abstracts* their use:\n"
            "- R1-R3: Used conceptually for temporary values, operands (cycled).\n"
            "- ACC (Accumulator): Conceptually holds results of computations or return values.\n"
            "Changes reflect *simulated* activity based on Python execution events (line, call, return).\n"
            "This is NOT a literal trace of hardware registers."
        )
        # Position text below the x-axis label
        ax.text(0.5, -0.35, explanation, transform=ax.transAxes, fontsize=9,
                ha='center', va='top', wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))


    def _create_cache_plot(self) -> None:
        """Create cache performance plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[1])
        steps = self._get_plot_steps()
        num_steps = len(steps)

        # Use cache history which records state at each access attempt
        hits_hist = self.cpu.cache.hits_history
        misses_hist = self.cpu.cache.misses_history

        # Align history length with number of recorded CPU steps
        if len(hits_hist) < num_steps:
            hits_hist.extend([hits_hist[-1]] * (num_steps - len(hits_hist)))
            misses_hist.extend([misses_hist[-1]] * (num_steps - len(misses_hist)))
        elif len(hits_hist) > num_steps:
            hits_hist = hits_hist[:num_steps]
            misses_hist = misses_hist[:num_steps]


        # Calculate hit rate at each step
        hit_rates = []
        total_accesses_hist = [h + m for h, m in zip(hits_hist, misses_hist)]
        for i in range(num_steps):
            total_accesses = total_accesses_hist[i]
            if total_accesses == 0:
                hit_rates.append(0) # Avoid division by zero, default to 0%
            else:
                hit_rates.append((hits_hist[i] / total_accesses) * 100)

        # Plot cumulative hits and misses over time
        ax.plot(steps, hits_hist, marker='.', linestyle='-', color='green', label=f'Cumulative Hits (Final: {hits_hist[-1]})')
        ax.plot(steps, misses_hist, marker='.', linestyle='-', color='red', label=f'Cumulative Misses (Final: {misses_hist[-1]})')

        # Plot hit rate on a secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps, hit_rates, marker='.', linestyle='--', color='blue', label='Hit Rate (%)')
        ax2.set_ylabel("Hit Rate (%)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 105) # Set Y limit for percentage

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='center left', fontsize='small')

        ax.set_title("Simulated Cache Performance", pad=15)
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Cumulative Count")
        ax.grid(True, linestyle=':')

        # Add embedded explanatory text
        explanation = (
            "Cache: Small, fast memory storing recently used data.\n"
            "- Cache Hit (Green): Data found in cache (fast). Triggered by simulated memory reads/writes.\n"
            "- Cache Miss (Red): Data NOT in cache (slower - requires fetching). \n"
            "- Hit Rate (Blue %): Efficiency = Hits / Total Accesses.\n"
            "This uses a simple Pseudo-LRU simulation on variable names as addresses."
        )
        ax.text(0.5, -0.40, explanation, transform=ax.transAxes, fontsize=9,
                ha='center', va='top', wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))


    def _create_memory_plot(self) -> None:
        """Create memory usage plot with educational annotations"""
        ax = self.fig.add_subplot(self.gs[2])
        steps = self._get_plot_steps()
        num_steps = len(steps)

        # Get history from memory component
        stack_hist = self.cpu.memory.stack_usage_history
        heap_hist = self.cpu.memory.heap_usage_history

        # Align history length with number of recorded CPU steps
        if len(stack_hist) < num_steps:
             stack_hist.extend([stack_hist[-1]] * (num_steps - len(stack_hist)))
             heap_hist.extend([heap_hist[-1]] * (num_steps - len(heap_hist)))
        elif len(stack_hist) > num_steps:
            stack_hist = stack_hist[:num_steps]
            heap_hist = heap_hist[:num_steps]

        # Create stacked area chart
        # Plot heap first, then stack on top
        total_usage = [s + h for s, h in zip(stack_hist, heap_hist)]
        ax.fill_between(steps, 0, heap_hist, alpha=0.6, color='seagreen', label=f'Simulated Heap ({heap_hist[-1]} allocs)')
        ax.fill_between(steps, heap_hist, total_usage, alpha=0.6, color='royalblue', label=f'Simulated Stack (Depth {stack_hist[-1]})')


        ax.set_title("Simulated Memory Usage", pad=15)
        ax.set_xlabel("Execution Step")
        ax.set_ylabel("Simulated Units (Stack Depth / Heap Allocations)")
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle=':')
        ax.set_ylim(bottom=0) # Ensure y-axis starts at 0


        # Add embedded explanatory text
        explanation = (
            "Memory Usage (Abstract Model):\n"
            "- Stack (Blue): Grows with function calls (depth), shrinks on return. Stores local context.\n"
            "- Heap (Green): Tracks simulated dynamic allocations (e.g., conceptual object creation).\n"
            "   Increases here are triggered by simulated writes or inferred object needs.\n"
            "Recursive functions show significant stack growth. Data-heavy algorithms might show more heap activity.\n"
            "This does NOT represent actual memory addresses or sizes in bytes."
        )
        ax.text(0.5, -0.45, explanation, transform=ax.transAxes, fontsize=9,
                ha='center', va='top', wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', fc='lavender', alpha=0.8))


# --- AI Assistant (Static Descriptions) ---

class AIAssistant:
    """Provides static educational descriptions of simulation results"""
    def __init__(self):
        # No API key needed for static descriptions
        pass

    def describe_visualization(self, plot_type: str) -> str:
        """Provide educational description for each plot type"""
        descriptions = {
            'registers': (
                "REGISTER ANALYSIS:\n"
                "The register plot *simulates* how the CPU might use its fastest storage. "
                "The ACC (Accumulator) often holds calculation results or return values. R1-R3 "
                "are shown holding temporary values based on simulated operations derived from "
                "Python's execution flow (calls, returns, lines). High activity can indicate "
                "complex calculations or loops. Remember, this is a conceptual model."
            ),
            'cache': (
                "CACHE ANALYSIS:\n"
                "This plot shows the efficiency of the *simulated* cache. High hit rates (data found in cache) "
                "are desirable for speed. Misses mean the CPU conceptually had to fetch data from slower "
                "main memory. The simulation triggers cache checks based on inferred variable access "
                "during line execution. Algorithms accessing data repeatedly or locally tend to have better hit rates."
            ),
            'memory': (
                "MEMORY ANALYSIS:\n"
                "Memory usage is abstractly tracked:\n"
                "- Stack: Shows function call depth. Deep recursion leads to high stack usage.\n"
                "- Heap: Represents simulated dynamic allocations. This might increase when the "
                "simulation infers data structures are being created or modified.\n"
                "Understanding how an algorithm uses stack vs. heap helps in analyzing its memory footprint "
                "and potential issues like stack overflows (in real systems)."
            )
        }
        return descriptions.get(plot_type, "No description available for this plot type.")

# --- Main Execution Logic ---

def load_algorithm(file_path: str):
    """Load algorithm module from file path"""
    if not os.path.exists(file_path):
        print(f"Error: Algorithm file '{file_path}' not found")
        sys.exit(1)

    try:
        module_name = "algorithm" # Fixed module name for import
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module # Register module
        spec.loader.exec_module(module)
        return module, os.path.abspath(file_path) # Return module and its absolute path
    except Exception as e:
        print(f"Error loading algorithm from '{file_path}': {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the CPU simulation"""
    if len(sys.argv) < 2:
        print("\nUsage: python main.py [algorithm_file.py] [arguments...]")
        print("Example (Factorial): python main.py factorial.py 5")
        print("Example (Sum List):  python main.py sum_list.py 1 2 3 4 5")
        print("\nSee docstring in main.py for details on the simulation model.\n")
        sys.exit(1)

    algorithm_file = sys.argv[1]
    algorithm_args_str = sys.argv[2:]
    algorithm_args = []

    # Parse arguments, attempting conversion to int/float first
    for arg in algorithm_args_str:
        try:
            algorithm_args.append(int(arg))
        except ValueError:
            try:
                algorithm_args.append(float(arg))
            except ValueError:
                algorithm_args.append(arg) # Keep as string if not numeric

    # Load the algorithm
    algorithm_module, algorithm_abs_path = load_algorithm(algorithm_file)

    # Initialize CPU and execute algorithm
    cpu = CPU()

    try:
        print(f"\nSimulating CPU execution for algorithm from '{algorithm_file}' with args: {algorithm_args}...")
        result = cpu.execute_algorithm(algorithm_module, algorithm_abs_path, *algorithm_args)
        print(f"Algorithm execution finished. Final Result: {result}")
        print(f"Total simulated steps: {cpu.step_count}")

        # Generate and display visualizations
        if cpu.execution_steps:
             print("\nGenerating CPU simulation visualizations...")
             visualizer = CPUVisualizer(cpu)
             visualizer.generate_visualizations()

             # Provide AI descriptions (static)
             ai_assistant = AIAssistant()
             print("\n=== EDUCATIONAL ANALYSIS ===")
             print(ai_assistant.describe_visualization('registers'))
             print("-" * 30)
             print(ai_assistant.describe_visualization('cache'))
             print("-" * 30)
             print(ai_assistant.describe_visualization('memory'))
             print("=" * 30)
        else:
             print("Simulation did not record any steps.")

    except Exception as e:
        print(f"\n--- Simulation Error ---")
        import traceback
        print(f"An error occurred during simulation: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("------------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()