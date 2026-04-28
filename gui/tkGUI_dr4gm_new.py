#!/usr/bin/env python3
"""
DR4GM GUI - Modern 6-Phase Workflow Interface
Based on the WORKFLOW.md 6-phase processing pipeline
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os, signal, sys
import threading
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import Image, ImageTk

matplotlib.use('TkAgg')

class DR4GMWorkflowGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DR4GM v1.0")
        self.geometry("1400x800")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        self.ultra_light_blue = "#E0F7FF"
        
        # Initialize paths
        self.input_dir = tk.StringVar(value=os.getcwd())
        self.output_dir = tk.StringVar(value="./results")
        self.current_phase = tk.StringVar(value="Phase 1: Data Conversion")
        
        # Processing parameters
        self.code_type = tk.StringVar(value="eqdyna")
        self.grid_resolution = tk.StringVar(value="1000")
        self.distance_range = tk.StringVar(value="0 30000")
        self.distance_bin_size = tk.StringVar(value="500")
        
        # Create GUI layout
        self.setup_gui()
        
        # Protocol for window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_gui(self):
        """Create the main GUI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main grid weights
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Left side (control panel) - 50%
        main_frame.columnconfigure(1, weight=1)  # Right side (canvas) - 50%
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls - constrained to its space
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0,5))
        control_frame.grid_propagate(False)  # Don't let children resize the frame
        
        # Right panel for visualization - constrained to its space  
        viz_frame = ttk.Frame(main_frame, padding="5")
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5,0))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.grid_propagate(False)  # Don't let children resize the frame
        
        # Setup left panel sections
        self.setup_path_section(control_frame)
        self.setup_workflow_section(control_frame)
        self.setup_viewer_section(control_frame)
        self.setup_execution_section(control_frame)
        
        # Setup right panel (canvas)
        self.setup_canvas(viz_frame)
    
    def setup_path_section(self, parent):
        """Setup file path selection section using dropdown navigation"""
        path_frame = ttk.LabelFrame(parent, text="Data Paths & Settings", padding="5")
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0,10))
        path_frame.columnconfigure(1, weight=1)
        
        # Input directory with dropdown navigation (like original GUI)
        ttk.Label(path_frame, text="Input Data Path:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_current_path = tk.StringVar(value=os.getcwd())
        self.input_display_path = tk.StringVar(value=self.truncate_path(os.getcwd()))
        self.input_navigate_button = tk.Menubutton(path_frame, textvariable=self.input_display_path, 
                                                  relief=tk.RAISED, anchor="w")
        self.input_navigate_button.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        self.input_navigate_button.menu = tk.Menu(self.input_navigate_button, tearoff=0)
        self.input_navigate_button["menu"] = self.input_navigate_button.menu
        self.update_dropdown_menu(self.input_navigate_button, self.input_current_path, 'input')
        
        # Output directory with dropdown navigation
        ttk.Label(path_frame, text="Output Data Path:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Ensure output directory exists
        default_output = os.path.join(os.getcwd(), "results")
        os.makedirs(default_output, exist_ok=True)
        
        self.output_current_path = tk.StringVar(value=default_output)
        self.output_display_path = tk.StringVar(value=self.truncate_path(default_output))
        self.output_navigate_button = tk.Menubutton(path_frame, textvariable=self.output_display_path, 
                                                   relief=tk.RAISED, anchor="w")
        self.output_navigate_button.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        self.output_navigate_button.menu = tk.Menu(self.output_navigate_button, tearoff=0)
        self.output_navigate_button["menu"] = self.output_navigate_button.menu
        
        # Initialize the output_dir variable
        self.output_dir.set(default_output)
        
        # Update both current path and display path
        self.output_display_path.set(self.truncate_path(default_output))
        self.update_dropdown_menu(self.output_navigate_button, self.output_current_path, 'output')
        
        # Processing parameters moved here
        ttk.Separator(path_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Simulation Code and Grid Resolution on one line
        params_frame1 = ttk.Frame(path_frame)
        params_frame1.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=1)
        params_frame1.columnconfigure(1, weight=1)
        params_frame1.columnconfigure(3, weight=1)
        
        ttk.Label(params_frame1, text="Code:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        code_combo = ttk.Combobox(params_frame1, textvariable=self.code_type, 
                                 values=["eqdyna", "fd3d", "waveqlab3d", "seissol", "sord", "mafe", "specfem3d"],
                                 state="readonly")
        code_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0,10))
        
        ttk.Label(params_frame1, text="Grid (m):").grid(row=0, column=2, sticky=tk.W, padx=(0,5))
        ttk.Entry(params_frame1, textvariable=self.grid_resolution).grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Distance Range and Bin Size on one line
        params_frame2 = ttk.Frame(path_frame)
        params_frame2.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=1)
        params_frame2.columnconfigure(1, weight=1)
        params_frame2.columnconfigure(3, weight=1)
        
        ttk.Label(params_frame2, text="Dist Range (m):").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        ttk.Entry(params_frame2, textvariable=self.distance_range).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0,10))
        
        ttk.Label(params_frame2, text="Bin Size (m):").grid(row=0, column=2, sticky=tk.W, padx=(0,5))
        ttk.Entry(params_frame2, textvariable=self.distance_bin_size).grid(row=0, column=3, sticky=(tk.W, tk.E))
    
    def setup_workflow_section(self, parent):
        """Setup workflow phase selection"""
        workflow_frame = ttk.LabelFrame(parent, text="DR4GM 6-Phase Workflow", padding="5")
        workflow_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0,10))
        workflow_frame.columnconfigure(0, weight=1)
        workflow_frame.columnconfigure(1, weight=1)
        
        # Phase buttons in 2 columns
        phases = [
            ("Phase 1: Data Conversion", self.run_phase1),
            ("Phase 2: Station Subset", self.run_phase2),
            ("Phase 3: GM Processing", self.run_phase3),
            ("Phase 4: Map Visualization", self.run_phase4),
            ("Phase 5: GM Statistics", self.run_phase5),
            ("Phase 6: Stats Visualization", self.run_phase6)
        ]
        
        for i, (phase_name, command) in enumerate(phases):
            row = i // 2
            col = i % 2
            ttk.Button(workflow_frame, text=phase_name, 
                      command=command).grid(row=row, column=col, sticky=(tk.W, tk.E), pady=2, padx=1)
        
        # Run all phases button
        ttk.Separator(workflow_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Button(workflow_frame, text="🚀 Run Complete Workflow", 
                  command=self.run_complete_workflow, 
                  style="Accent.TButton").grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def setup_viewer_section(self, parent):
        """Setup unified viewer controls for maps and visualizations"""
        viewer_frame = ttk.LabelFrame(parent, text="Viewer", padding="5")
        viewer_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0,10))
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.columnconfigure(2, weight=1)
        
        # Row 1: Maps dropdown
        ttk.Label(viewer_frame, text="Maps:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.selected_map = tk.StringVar(value="Select map...")
        self.map_dropdown = ttk.Combobox(viewer_frame, textvariable=self.selected_map, 
                                        state="readonly", width=20)
        self.map_dropdown.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=2)
        self.map_dropdown.bind('<<ComboboxSelected>>', self.on_map_selected)
        
        # Row 2: Visualization buttons in 3 columns
        viz_buttons = [
            ("GM Maps", self.show_gm_maps),
            ("Attenuation", self.show_attenuation_plots),
            ("Response Spectra", self.show_response_spectra)
        ]
        
        for i, (button_name, command) in enumerate(viz_buttons):
            ttk.Button(viewer_frame, text=button_name, 
                      command=command).grid(row=1, column=i, sticky=(tk.W, tk.E), pady=2, padx=2)
        
        # Row 3: Statistics and Clear buttons
        ttk.Button(viewer_frame, text="Statistics", 
                  command=self.show_statistics).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2, padx=2)
        ttk.Button(viewer_frame, text="Clear Canvas", 
                  command=self.clear_canvas).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=2)
        
        # Update available maps when output directory changes
        self.update_available_maps()
    
    def setup_execution_section(self, parent):
        """Setup execution controls"""
        exec_frame = ttk.LabelFrame(parent, text="Execution Status", padding="5")
        exec_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0,10))
        exec_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress = ttk.Progressbar(exec_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status text
        self.status_text = tk.Text(exec_frame, height=4, width=50)
        scrollbar = ttk.Scrollbar(exec_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        exec_frame.rowconfigure(1, weight=1)
    
    def setup_canvas(self, parent):
        """Setup matplotlib canvas for visualization"""
        canvas_frame = ttk.LabelFrame(parent, text="Visualization Canvas", padding="5")
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add navigation toolbar
        toolbar = ttk.Frame(canvas_frame)
        toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar)
        self.toolbar.update()
    
    # Directory navigation methods (unified from original GUI)
    def getDirectoryContents(self, path):
        try:
            contents = os.listdir(path)
            contents.insert(0, '..')
            return contents
        except FileNotFoundError:
            return ['..']
    
    def update_dropdown_menu(self, button, current_path_var, path_type):
        """Unified dropdown menu update for both input and output paths"""
        button.menu.delete(0,'end')
        current_dir = current_path_var.get()
        
        # For output paths, ensure directory exists
        if path_type == 'output' and not os.path.exists(current_dir):
            try:
                os.makedirs(current_dir, exist_ok=True)
            except:
                # If we can't create it, go to parent directory
                current_dir = os.path.dirname(current_dir)
                current_path_var.set(current_dir)
                # Update display path as well
                if path_type == 'output':
                    self.output_display_path.set(self.truncate_path(current_dir))
                else:
                    self.input_display_path.set(self.truncate_path(current_dir))
        
        contents = self.getDirectoryContents(current_dir)
            
        for item in contents:
            button.menu.add_command(
                label=item,
                command = lambda item=item, path_type=path_type: self.changeDirectory(item, path_type)
            )
    
    def changeDirectory(self, item, path_type):
        """Unified directory change function (from original GUI)"""
        if path_type == 'input':
            current_path_var = self.input_current_path
            display_path_var = self.input_display_path
            target_var = self.input_dir
            button = self.input_navigate_button
        else:  # output
            current_path_var = self.output_current_path
            display_path_var = self.output_display_path
            target_var = self.output_dir
            button = self.output_navigate_button
            
        if item == "..":
            new_path = os.path.dirname(current_path_var.get())
        else:
            new_path = os.path.join(current_path_var.get(), item)

        # For input paths, directory must exist
        if path_type == 'input':
            if os.path.isdir(new_path):
                current_path_var.set(new_path)
                display_path_var.set(self.truncate_path(new_path))
                target_var.set(new_path)
                self.update_dropdown_menu(button, current_path_var, path_type)
                self.log_message(f"Input path set to: {new_path}")
        else:
            # For output paths, create if needed
            if os.path.isdir(new_path):
                current_path_var.set(new_path)
                display_path_var.set(self.truncate_path(new_path))
                target_var.set(new_path)
                self.update_dropdown_menu(button, current_path_var, path_type)
                self.log_message(f"Output path set to: {new_path}")
            else:
                try:
                    if item != "..":  # Don't create parent directories
                        os.makedirs(new_path, exist_ok=True)
                        current_path_var.set(new_path)
                        display_path_var.set(self.truncate_path(new_path))
                        target_var.set(new_path)
                        self.update_dropdown_menu(button, current_path_var, path_type)
                        self.log_message(f"Created and set output path to: {new_path}")
                    else:
                        # Navigate to parent
                        current_path_var.set(new_path)
                        display_path_var.set(self.truncate_path(new_path))
                        target_var.set(new_path)
                        self.update_dropdown_menu(button, current_path_var, path_type)
                        self.log_message(f"Output path set to: {new_path}")
                except Exception as e:
                    self.log_message(f"Could not create output directory: {new_path} - {str(e)}")
    
    # File dialog methods (backup options)
    def browse_input_dir(self):
        directory = filedialog.askdirectory(initialdir=self.input_dir.get())
        if directory:
            self.input_dir.set(directory)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
    
    # Phase execution methods
    def run_phase1(self):
        """Phase 1: Data Conversion"""
        self.current_phase.set("Phase 1: Data Conversion")
        
        # Get the selected paths and code type
        input_path = self.input_dir.get()
        output_path = self.output_dir.get()
        code_type = self.code_type.get()
        
        # Validate selections
        if not input_path or not os.path.isdir(input_path):
            self.log_message("ERROR: Please select a valid input directory")
            return
        
        if not code_type:
            self.log_message("ERROR: Please select a simulation code type")
            return
        
        self.log_message(f"Starting Phase 1: Data Conversion...")
        self.log_message(f"  Input: {input_path}")
        self.log_message(f"  Output: {output_path}")
        self.log_message(f"  Code: {code_type}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Build command string 
        converter_script = f"{code_type}_converter_api.py"
        cmd = f"python {converter_script} --input_dir {input_path} --output_dir {output_path} --verbose"
        
        # Run command in background thread
        self.run_command_async(cmd, "Phase 1 completed!")
    
    def run_phase2(self):
        """Phase 2: Station Subset Selection"""
        self.current_phase.set("Phase 2: Station Subset Selection")
        self.log_message("Starting Phase 2: Station Subset Selection...")
        
        velocities_file = os.path.join(self.output_dir.get(), "velocities.npz")
        output_file = os.path.join(self.output_dir.get(), f"grid_{self.grid_resolution.get()}m.npz")
        
        cmd = f"python station_subset_selector.py --input_npz {velocities_file} --output_npz {output_file} --grid_resolution {self.grid_resolution.get()}"
        
        self.run_command_async(cmd, "Phase 2 completed!")
    
    def run_phase3(self):
        """Phase 3: Ground Motion Processing"""
        self.current_phase.set("Phase 3: Ground Motion Processing")
        self.log_message("Starting Phase 3: Ground Motion Processing...")
        
        grid_file = os.path.join(self.output_dir.get(), f"grid_{self.grid_resolution.get()}m.npz")
        
        cmd = f"python npz_gm_processor.py --velocity_npz {grid_file} --output_dir {self.output_dir.get()}"
        
        self.run_command_async(cmd, "Phase 3 completed!")
    
    def run_phase4(self):
        """Phase 4: Map Visualization"""
        self.current_phase.set("Phase 4: Map Visualization")
        self.log_message("Starting Phase 4: Map Visualization...")
        
        gm_file = os.path.join(self.output_dir.get(), "ground_motion_metrics.npz")
        
        cmd = f"python visualize_gm_maps.py --gm_npz {gm_file} --output_dir {self.output_dir.get()}"
        
        # Custom async run with callback to update maps dropdown
        def run():
            self.progress.start()
            try:
                # Change to utils directory for command execution
                original_dir = os.getcwd()
                utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                os.chdir(utils_dir)
                
                # Handle both string and list commands
                if isinstance(cmd, str):
                    cmd_str = cmd
                    cmd_list = cmd.split()
                else:
                    cmd_str = ' '.join(cmd)
                    cmd_list = cmd
                
                print(f"[GUI] Executing command: {cmd_str}")
                self.log_message(f"Executing: {cmd_str}")
                self.log_message(f"Working directory: {utils_dir}")
                self.log_message("-" * 50)
                
                process = subprocess.Popen(
                    cmd_list, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    universal_newlines=True
                )
                
                # Stream output to GUI
                for line in iter(process.stdout.readline, ''):
                    if line.strip():  # Only log non-empty lines
                        self.log_message(line.strip())
                
                process.wait()
                
                print(f"[GUI] Command finished with return code: {process.returncode}")
                self.log_message("-" * 50)
                
                if process.returncode == 0:
                    self.log_message("Phase 4 completed!")
                    print("[GUI] Phase 4 completed!")
                    # Update available maps after successful completion
                    self.update_available_maps()
                    self.log_message("Maps dropdown updated with new visualizations.")
                else:
                    error_msg = f"Command failed with return code {process.returncode}"
                    self.log_message(error_msg)
                    print(f"[GUI] {error_msg}")
                
                os.chdir(original_dir)
                
            except Exception as e:
                error_msg = f"Error executing command: {str(e)}"
                self.log_message(error_msg)
                print(f"[GUI] {error_msg}")
            finally:
                self.progress.stop()
        
        import threading
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def run_phase5(self):
        """Phase 5: Ground Motion Statistics"""
        self.current_phase.set("Phase 5: Ground Motion Statistics")
        self.log_message("Starting Phase 5: Ground Motion Statistics...")
        
        gm_file = os.path.join(self.output_dir.get(), "ground_motion_metrics.npz")
        distance_min, distance_max = self.distance_range.get().split()
        
        cmd = f"python gm_stats.py --gm_data {gm_file} --output_dir {self.output_dir.get()} --distance_range {distance_min} {distance_max} --distance_bin_size {self.distance_bin_size.get()}"
        
        self.run_command_async(cmd, "Phase 5 completed!")
    
    def run_phase6(self):
        """Phase 6: Statistics Visualization"""
        self.current_phase.set("Phase 6: Statistics Visualization")
        self.log_message("Starting Phase 6: Statistics Visualization...")
        
        stats_file = os.path.join(self.output_dir.get(), "gm_statistics.npz")
        
        cmd = f"python visualize_gm_stats.py --stats_data {stats_file} --output_dir {self.output_dir.get()} --verbose"
        
        self.run_command_async(cmd, "Phase 6 completed!")
    
    def run_complete_workflow(self):
        """Run all 6 phases in sequence"""
        self.log_message("Starting complete DR4GM workflow...")
        
        cmd = f"./run_all.sh {self.input_dir.get()} {self.code_type.get()} {self.output_dir.get()} {self.grid_resolution.get()}"
        
        self.run_command_async(cmd, "Complete workflow finished!")
    
    # Visualization methods
    def show_gm_maps(self):
        """Display ground motion maps"""
        self.log_message("Loading ground motion maps...")
        output_dir = self.output_dir.get()
        
        # Look for common map files
        map_files = []
        for metric in ['PGA', 'PGV', 'PGD', 'CAV']:
            map_file = os.path.join(output_dir, f"{metric}_map.png")
            if os.path.exists(map_file):
                map_files.append((metric, map_file))
        
        if map_files:
            self.display_multiple_images(map_files, "Ground Motion Maps")
        else:
            self.log_message("No ground motion maps found. Run Phase 4 first.")
    
    def show_attenuation_plots(self):
        """Display attenuation plots"""
        self.log_message("Loading attenuation plots...")
        output_dir = self.output_dir.get()
        
        # Look for attenuation plot files with correct naming pattern
        import glob
        plot_files = []
        
        # Look for basic GM metrics vs distance plots
        for metric in ['PGA', 'PGV', 'PGD', 'CAV']:
            pattern = os.path.join(output_dir, f"gm{metric}StatsVsR.png")
            matches = glob.glob(pattern)
            for match in matches:
                plot_files.append((metric, match))
        
        # Look for RSA vs distance plots
        rsa_pattern = os.path.join(output_dir, "gmRSA_T_*StatsVsR.png")
        rsa_matches = glob.glob(rsa_pattern)
        for match in rsa_matches:
            # Extract period from filename like gmRSA_T_1_000StatsVsR.png
            filename = os.path.basename(match)
            period_part = filename.replace("gmRSA_T_", "").replace("StatsVsR.png", "")
            period = period_part.replace("_", ".")
            plot_files.append((f"RSA T={period}s", match))
        
        if plot_files:
            self.display_multiple_images(plot_files, "Attenuation Plots")
        else:
            self.log_message("No attenuation plots found. Run Phase 6 first.")
    
    def show_response_spectra(self):
        """Display response spectra plots"""
        self.log_message("Loading response spectra plots...")
        output_dir = self.output_dir.get()
        
        # Look for RSA plot files
        rsa_files = []
        rsa_pattern = "RSA_T"
        
        for file in os.listdir(output_dir):
            if file.startswith(rsa_pattern) and file.endswith('.png'):
                period = file.replace('RSA_T', '').replace('s_vs_distance.png', '').replace('_', '.')
                rsa_files.append((f"T={period}s", os.path.join(output_dir, file)))
        
        if rsa_files:
            self.display_multiple_images(rsa_files[:4], "Response Spectra")  # Show first 4
        else:
            self.log_message("No response spectra plots found. Run Phase 6 first.")
    
    def show_statistics(self):
        """Display statistics summary"""
        self.log_message("Loading statistics summary...")
        
        # Create a simple statistics summary plot
        stats_file = os.path.join(self.output_dir.get(), "gm_statistics.npz")
        if os.path.exists(stats_file):
            self.create_stats_summary(stats_file)
        else:
            self.log_message("No statistics file found. Run Phase 5 first.")
    
    def display_multiple_images(self, image_files, title):
        """Display multiple images in subplots"""
        self.fig.clear()
        
        n_images = len(image_files)
        if n_images == 0:
            return
        
        # Calculate subplot layout
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols
        
        for i, (name, filepath) in enumerate(image_files):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            
            try:
                img = Image.open(filepath)
                ax.imshow(img)
                ax.set_title(name, fontsize=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(name, fontsize=10)
        
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def create_stats_summary(self, stats_file):
        """Create a statistics summary plot"""
        try:
            import numpy as np
            
            data = np.load(stats_file)
            self.fig.clear()
            
            ax = self.fig.add_subplot(111)
            
            # Plot a simple summary - distance vs PGA mean
            if 'rjb_distance_bins' in data and 'PGA_mean' in data:
                distances = data['rjb_distance_bins'] / 1000  # Convert to km
                pga_mean = data['PGA_mean'] / 981.0  # Convert to g
                
                ax.loglog(distances, pga_mean, 'o-', label='PGA Mean')
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('PGA (g)')
                ax.set_title('Ground Motion Statistics Summary')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Statistics data format not recognized", 
                       ha='center', va='center', transform=ax.transAxes)
            
            self.canvas.draw()
            self.log_message("Statistics summary displayed.")
            
        except Exception as e:
            self.log_message(f"Error creating statistics summary: {str(e)}")
    
    # Map viewing methods
    def update_available_maps(self):
        """Update the dropdown with available map files"""
        output_dir = self.output_dir.get()
        if not output_dir or not os.path.exists(output_dir):
            self.map_dropdown['values'] = ["No maps available"]
            return
        
        # Look for common map files generated by Phase 4
        map_patterns = [
            "*summary*.png",
            "*PGA*.png", 
            "*PGV*.png",
            "*PGD*.png", 
            "*RSA*.png",
            "*CAV*.png"
        ]
        
        available_maps = []
        for pattern in map_patterns:
            import glob
            maps = glob.glob(os.path.join(output_dir, pattern))
            for map_file in maps:
                map_name = os.path.basename(map_file)
                available_maps.append(map_name)
        
        if available_maps:
            self.map_dropdown['values'] = sorted(available_maps)
            self.selected_map.set("Select map...")
        else:
            self.map_dropdown['values'] = ["No maps found - run Phase 4 first"]
            self.selected_map.set("No maps found - run Phase 4 first")
    
    def on_map_selected(self, event=None):
        """Auto-display selected map on the canvas when dropdown selection changes"""
        selected = self.selected_map.get()
        if not selected or selected in ["Select map...", "No maps available", "No maps found - run Phase 4 first"]:
            return
        
        output_dir = self.output_dir.get()
        map_path = os.path.join(output_dir, selected)
        
        if not os.path.exists(map_path):
            self.log_message(f"Map file not found: {map_path}")
            return
        
        try:
            from PIL import Image
            # Load and display the image
            img = Image.open(map_path)
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(selected, fontsize=12)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.log_message(f"Displaying: {selected}")
            
        except Exception as e:
            self.log_message(f"Error loading map {selected}: {str(e)}")
    
    def clear_canvas(self):
        """Clear the visualization canvas"""
        self.fig.clear()
        self.canvas.draw()
        self.log_message("Canvas cleared.")
    
    # Utility methods
    def truncate_path(self, path, max_length=50):
        """Truncate long paths for display in GUI"""
        if len(path) <= max_length:
            return path
        
        # Split path and show beginning and end
        parts = path.split(os.sep)
        if len(parts) <= 2:
            return path
        
        # Show first part and last few parts
        result = parts[0] + os.sep + "..." + os.sep + os.sep.join(parts[-2:])
        
        # If still too long, just truncate
        if len(result) > max_length:
            return "..." + path[-(max_length-3):]
        
        return result

    def run_command_async(self, cmd, success_message):
        """Run a command in a separate thread"""
        def run():
            self.progress.start()
            try:
                # Change to utils directory for command execution
                original_dir = os.getcwd()
                utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                os.chdir(utils_dir)
                
                # Handle both string and list commands
                if isinstance(cmd, str):
                    cmd_str = cmd
                    cmd_list = cmd.split()
                else:
                    cmd_str = ' '.join(cmd)
                    cmd_list = cmd
                
                print(f"[GUI] Executing command: {cmd_str}")
                self.log_message(f"Executing: {cmd_str}")
                self.log_message(f"Working directory: {utils_dir}")
                self.log_message("-" * 50)
                
                process = subprocess.Popen(
                    cmd_list, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    universal_newlines=True
                )
                
                # Stream output to GUI
                for line in iter(process.stdout.readline, ''):
                    if line.strip():  # Only log non-empty lines
                        self.log_message(line.strip())
                
                process.wait()
                
                print(f"[GUI] Command finished with return code: {process.returncode}")
                self.log_message("-" * 50)
                
                if process.returncode == 0:
                    self.log_message(success_message)
                    print(f"[GUI] {success_message}")
                else:
                    error_msg = f"Command failed with return code {process.returncode}"
                    self.log_message(error_msg)
                    print(f"[GUI] {error_msg}")
                
                os.chdir(original_dir)
                
            except Exception as e:
                error_msg = f"Error executing command: {str(e)}"
                self.log_message(error_msg)
                print(f"[GUI] {error_msg}")
            finally:
                self.progress.stop()
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def log_message(self, message):
        """Add message to status text widget"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.update()
    
    def on_closing(self):
        """Handle window closing"""
        self.quit()
        self.destroy()
        sys.exit()

def signal_handler(sig, frame):
    print('Exiting. You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        app = DR4GMWorkflowGUI()
        app.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
        sys.exit(0)