#!/usr/bin/env python3
import sys
import os
import importlib.util
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from typing import List, Dict, Any, Tuple

# CPU Simulation classes
class Register:
    """Simulates a CPU register"""
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
    """Simulates a simple CPU cache"""
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
                # If cache is full, remove oldest entry
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
    """Simulates a CPU executing an algorithm"""
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
        """Execute the provided algorithm and simulate CPU operations"""
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
        # Create a dictionary to track local variables
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Store function arguments in registers and stack
        for i, (arg_name, arg_value) in enumerate(zip(arg_names, args)):
            reg_name = f"R{i+1}" if f"R{i+1}" in self.registers else "ACC"
            self.registers[reg_name].set(arg_value)
            self.memory.stack_allocate(arg_name, arg_value)
            self._record_step(f"Initialize {reg_name} with argument {arg_name}={arg_value}")
        
        # For factorial example: simulate multiplication steps
        if func.__name__ == "factorial":
            n = args[0]
            result = 1
            self.registers["ACC"].set(result)
            
            # Simulate factorial calculation
            for i in range(1, n + 1):
                # Access variable from cache/memory
                self._record_step(f"Calculate factorial step: {i}")
                
                # Load value into registers
                self.registers["R1"].set(i)
                self._record_step(f"Load {i} into R1")
                
                # Get current result from accumulator
                current_result = self.registers["ACC"].get()
                self._record_step(f"Access current result {current_result} from ACC")
                
                # Simulate cache operations
                cache_address = f"temp_{i}"
                self.cache.access(cache_address, i)
                
                # Perform multiplication
                new_result = current_result * i
                self._record_step(f"Multiply ACC({current_result}) * R1({i}) = {new_result}")
                
                # Store result back to accumulator
                self.registers["ACC"].set(new_result)
                self._record_step(f"Store result {new_result} in ACC")
                
                # Simulate memory allocation for intermediate result
                self.memory.heap_allocate(f"result_{i}", new_result)
                
                # Add some simulated delay
                time.sleep(0.1)
            
            # Get final result
            result = self.registers["ACC"].get()
        else:
            # For other algorithms, just run the function and track the result
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
    """Handles visualization of CPU simulation results"""
    def __init__(self, cpu: CPU):
        self.cpu = cpu
        self.images = []
    
    def generate_visualizations(self) -> List[str]:
        """Generate and save all visualizations, return file paths"""
        self._create_register_plot()
        self._create_cache_plot()
        self._create_memory_plot()
        return self.images
    
    def _create_register_plot(self) -> None:
        """Create and save register values plot"""
        plt.figure(figsize=(10, 6))
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        
        for reg_name, register in self.cpu.registers.items():
            # Ensure history is the right length
            if len(register.history) < len(steps):
                register.history.extend([register.history[-1]] * (len(steps) - len(register.history)))
            plt.plot(steps, register.history[:len(steps)], marker='o', label=reg_name)
        
        plt.title("Register Values During Execution")
        plt.xlabel("Execution Step")
        plt.ylabel("Register Value")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = "register_values.png"
        plt.savefig(filename)
        plt.close()
        self.images.append(filename)
    
    def _create_cache_plot(self) -> None:
        """Create and save cache performance plot"""
        plt.figure(figsize=(10, 6))
        
        # Extract cache data
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        hits = [step["cache_hits"] for step in self.cpu.execution_steps]
        misses = [step["cache_misses"] for step in self.cpu.execution_steps]
        
        # Calculate hit rate for each step
        hit_rates = []
        for i, (hit, miss) in enumerate(zip(hits, misses)):
            if hit + miss == 0:
                hit_rates.append(0)
            else:
                hit_rates.append(hit / (hit + miss) * 100)
        
        # Plot hit rate
        plt.plot(steps, hit_rates, marker='o', color='green', label='Cache Hit Rate (%)')
        
        # Add cumulative hits and misses
        plt.bar([1], [hits[-1]], width=0.4, alpha=0.6, color='blue', label=f'Cache Hits ({hits[-1]})')
        plt.bar([2], [misses[-1]], width=0.4, alpha=0.6, color='red', label=f'Cache Misses ({misses[-1]})')
        
        plt.title("Cache Performance")
        plt.xlabel("Execution")
        plt.ylabel("Hit Rate (%)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = "cache_performance.png"
        plt.savefig(filename)
        plt.close()
        self.images.append(filename)
    
    def _create_memory_plot(self) -> None:
        """Create and save memory usage plot"""
        plt.figure(figsize=(10, 6))
        
        # Extract memory data
        steps = list(range(1, len(self.cpu.execution_steps) + 1))
        stack_usage = [step["stack_size"] for step in self.cpu.execution_steps]
        heap_usage = [step["heap_size"] for step in self.cpu.execution_steps]
        
        # Create stacked area chart
        plt.fill_between(steps, 0, stack_usage, alpha=0.5, color='blue', label='Stack Usage')
        plt.fill_between(steps, stack_usage, [s + h for s, h in zip(stack_usage, heap_usage)], 
                         alpha=0.5, color='green', label='Heap Usage')
        
        plt.title("Memory Usage During Execution")
        plt.xlabel("Execution Step")
        plt.ylabel("Number of Allocations")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        filename = "memory_usage.png"
        plt.savefig(filename)
        plt.close()
        self.images.append(filename)

class AIAssistant:
    """Integrates with AI API to describe visualizations"""
    def __init__(self):
        self.api_key = None  # TODO: Insert your ChatGPT 4o API key here
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def describe_visualization(self, image_path: str) -> str:
        """Send visualization to AI API and get description"""
        if not self.api_key:
            return "API key not provided. Please add your ChatGPT 4o API key to use this feature."
        
        # This is a placeholder for the actual API call
        # In a real implementation, you would:
        # 1. Convert the image to base64
        # 2. Create a proper prompt
        # 3. Send a request to the API with the image
        
        try:
            # Simulated API call (replace with actual implementation)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # This is a simplified example - in reality, you'd encode the image
            # and include it properly according to the OpenAI API specs
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an expert in computer architecture."},
                    {"role": "user", "content": f"Describe what's happening in this CPU simulation visualization: {image_path}"}
                ]
            }
            
            # For demonstration, we'll return a placeholder response
            # In a real implementation, you would uncomment the following:
            # response = requests.post(self.api_url, headers=headers, json=payload)
            # return response.json()["choices"][0]["message"]["content"]
            
            # Placeholder responses based on image type
            if "register" in image_path:
                return "This visualization shows how register values change during algorithm execution. The increasing values in the Accumulator (ACC) register indicate the factorial calculation is building up the result with each iteration, while other registers store intermediate values and inputs."
            elif "cache" in image_path:
                return "The cache performance visualization shows the hit rate improving as the algorithm progresses. Initially, we see cache misses as data is loaded for the first time, but subsequent accesses benefit from the cache, improving execution efficiency."
            elif "memory" in image_path:
                return "This memory usage chart shows how the algorithm allocates memory in both stack and heap regions. The stack stores local variables and function parameters, while the heap shows dynamic allocations for intermediate results. The gradual increase in heap usage indicates the algorithm is storing more data as it progresses."
            
        except Exception as e:
            return f"Error getting AI description: {str(e)}"

def load_algorithm(file_path: str):
    """Load algorithm module from file path"""
    if not os.path.exists(file_path):
        print(f"Error: Algorithm file '{file_path}' not found")
        sys.exit(1)
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("algorithm", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading algorithm: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the CPU simulation"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py [algorithm_file] [algorithm_args]")
        print("Example: python main.py factorial.py 5")
        sys.exit(1)
    
    algorithm_file = sys.argv[1]
    algorithm_args = []
    
    # Parse additional arguments if provided
    for arg in sys.argv[2:]:
        try:
            # Convert to int if possible
            algorithm_args.append(int(arg))
        except ValueError:
            # Otherwise keep as string
            algorithm_args.append(arg)
    
    # Load the algorithm
    algorithm_module = load_algorithm(algorithm_file)
    
    # Initialize CPU
    cpu = CPU()
    
    # Execute the algorithm with simulation
    try:
        print(f"\nExecuting algorithm from {algorithm_file} with args {algorithm_args}...")
        result = cpu.execute_algorithm(algorithm_module, *algorithm_args)
        print(f"Algorithm execution completed. Result: {result}\n")
        
        # Generate visualizations
        print("Generating CPU simulation visualizations...")
        visualizer = CPUVisualizer(cpu)
        image_paths = visualizer.generate_visualizations()
        
        # Get AI descriptions
        ai_assistant = AIAssistant()
        
        # Display results
        print("\n=== CPU SIMULATION RESULTS ===\n")
        
        for image_path in image_paths:
            print(f"\n📊 {image_path.replace('.png', '').replace('_', ' ').title()}")
            print("-" * 50)
            
            # Get AI description
            ai_description = ai_assistant.describe_visualization(image_path)
            print(f"\n🤖 AI Analysis:")
            print(ai_description)
            print("\n")
    
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()