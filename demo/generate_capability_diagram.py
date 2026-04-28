#!/usr/bin/env python3
"""
DR4GM Capability Diagram Generator

This script creates professional diagrams showing DR4GM's capabilities
for stakeholders and customers. Easily customizable for different audiences.

Usage:
    python generate_capability_diagram.py
    python generate_capability_diagram.py --style business
    python generate_capability_diagram.py --style technical
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import argparse
from pathlib import Path

class DR4GMDiagramGenerator:
    def __init__(self, style='comprehensive'):
        self.style = style
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Deep magenta
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red-orange
            'info': '#4B5563',         # Dark gray
            'background': '#F9FAFB',   # Light gray
            'text': '#1F2937',         # Dark text
            'light_blue': '#EFF6FF',   # Very light blue
            'light_green': '#F0FDF4',  # Very light green
            'light_orange': '#FFF7ED'  # Very light orange
        }
        
        # Set consistent styling
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.weight': 'normal',
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'figure.titlesize': 20
        })
        
    def create_comprehensive_diagram(self):
        """Create comprehensive capability overview diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Large, bold title
        title_box = FancyBboxPatch((5, 85), 90, 12, boxstyle="round,pad=1", 
                                  facecolor=self.colors['primary'], alpha=0.95, edgecolor='none')
        ax.add_patch(title_box)
        ax.text(50, 91, 'DR4GM', ha='center', va='center', fontsize=32, fontweight='bold', color='white')
        ax.text(50, 87, 'Streamline earthquake physics from high-fidelity dynamic ruptures into seismic hazard insights', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
        
        # Extended 3-step workflow - much taller cards, minimal white space
        workflow_y = 60  # Much higher to reduce white space
        box_height = 40  # Much taller
        box_width = 28
        
        # Step 1: Input - much taller with bullet points
        input_box = FancyBboxPatch((5, workflow_y-20), box_width, box_height, 
                                  boxstyle="round,pad=1", 
                                  facecolor=self.colors['primary'], alpha=0.9, edgecolor='none')
        ax.add_patch(input_box)
        ax.text(19, workflow_y+15, '1. INPUT', ha='center', va='center', 
               fontsize=20, fontweight='bold', color='white')
        ax.text(19, workflow_y+10, 'Dense source process and seismograms ', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(19, workflow_y+7, 'from dynamic ruptures', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        # Bullet points for Input
        ax.text(19, workflow_y-0, '• EQDyna', ha='center', va='center', fontsize=16, color='white')
        ax.text(19, workflow_y-3, '• FD3D', ha='center', va='center', fontsize=16, color='white')
        ax.text(19, workflow_y-6, '• SeisSol', ha='center', va='center', fontsize=16, color='white')
        ax.text(19, workflow_y-9, '• SORD', ha='center', va='center', fontsize=16, color='white')
        ax.text(19, workflow_y-12, '• SpecFem3D', ha='center', va='center', fontsize=16, color='white')
        ax.text(19, workflow_y-15, '• WaveQLab3D', ha='center', va='center', fontsize=16, color='white')
        # Step 2: Process
        process_box = FancyBboxPatch((36, workflow_y-20), box_width, box_height, 
                                    boxstyle="round,pad=1", 
                                    facecolor=self.colors['accent'], alpha=0.9, edgecolor='none')
        ax.add_patch(process_box)
        ax.text(50, workflow_y+15, '2. PROCESS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color='white')
        ax.text(50, workflow_y+10, 'Standards & Automated Pipeline', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white')
        
        # Bullet points for Process - much larger fonts
        ax.text(50, workflow_y+2, '• Standardized Format Conversion', ha='center', va='center', fontsize=16, color='white')
        ax.text(50, workflow_y-2, '• Quick Station Selection', ha='center', va='center', fontsize=16, color='white')
        ax.text(50, workflow_y-6, '• Ground Motion Analysis', ha='center', va='center', fontsize=16, color='white')
        ax.text(50, workflow_y-10, '• Statistical Processing', ha='center', va='center', fontsize=16, color='white')
        
        # Step 3: Products (changed from Results)
        output_box = FancyBboxPatch((67, workflow_y-20), box_width, box_height, 
                                   boxstyle="round,pad=1", 
                                   facecolor=self.colors['success'], alpha=0.9, edgecolor='none')
        ax.add_patch(output_box)
        ax.text(81, workflow_y+15, '3. PRODUCTS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color='white')
        ax.text(81, workflow_y+10, 'Decision Ready', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white')
        
        # Bullet points for Products - removed GUI and web interface, larger fonts
        ax.text(81, workflow_y+4, '• Statistical Reports', ha='center', va='center', fontsize=16, color='white')
        ax.text(81, workflow_y, '• Interactive Visualization Maps', ha='center', va='center', fontsize=16, color='white')
        ax.text(81, workflow_y-4, '• Standardized Datasets', ha='center', va='center', fontsize=16, color='white')
        ax.text(81, workflow_y-8, '• Analysis Tools', ha='center', va='center', fontsize=16, color='white')
        
        # Large arrows
        arrow1 = ConnectionPatch((33, workflow_y), (36, workflow_y), "data", "data",
                               arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=30, 
                               fc='black', ec='black', linewidth=6)
        ax.add_patch(arrow1)
        
        arrow2 = ConnectionPatch((64, workflow_y), (67, workflow_y), "data", "data",
                               arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=30, 
                               fc='black', ec='black', linewidth=6)
        ax.add_patch(arrow2)
        
        # Bottom section - bullet points with larger box, moved up a bit
        cta_box = FancyBboxPatch((5, 25), 90, 12, boxstyle="round,pad=1", 
                               facecolor=self.colors['secondary'], alpha=0.95, edgecolor='none')
        ax.add_patch(cta_box)
        ax.text(50, 32, '• Graphic User Interface', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
        ax.text(50, 28, '• Interactive website: https://dr4gm-web-explorer.streamlit.app/', 
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def create_technical_flowchart(self):
        """Create technical processing flowchart"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Title
        title_box = FancyBboxPatch((15, 90), 70, 8, boxstyle="round,pad=1", 
                                  facecolor=self.colors['primary'], alpha=0.9, edgecolor='none')
        ax.add_patch(title_box)
        ax.text(50, 94, 'DR4GM Technical Processing Workflow', 
                ha='center', va='center', fontsize=20, fontweight='bold', color='white')
        
        # Define processing steps with better styling
        steps = [
            {'name': 'Raw Data Input', 'y': 78, 'color': self.colors['light_blue'], 'border': self.colors['primary']},
            {'name': 'Format Conversion', 'y': 66, 'color': self.colors['light_orange'], 'border': self.colors['accent']},
            {'name': 'Station Subsetting', 'y': 54, 'color': self.colors['light_green'], 'border': self.colors['success']},
            {'name': 'GM Metric Calculation', 'y': 42, 'color': self.colors['light_blue'], 'border': self.colors['info']},
            {'name': 'Map Generation', 'y': 30, 'color': self.colors['light_orange'], 'border': self.colors['secondary']},
            {'name': 'Statistical Analysis', 'y': 18, 'color': self.colors['light_green'], 'border': self.colors['primary']},
            {'name': 'Visualization & Export', 'y': 6, 'color': self.colors['light_blue'], 'border': self.colors['accent']}
        ]
        
        # Draw workflow boxes - larger and more prominent
        for i, step in enumerate(steps):
            # Main process box
            box = FancyBboxPatch((35, step['y']-4), 30, 10, boxstyle="round,pad=1", 
                               facecolor=step['color'], edgecolor=step['border'], linewidth=3)
            ax.add_patch(box)
            ax.text(50, step['y'], step['name'], ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=step['border'])
            
            # Add arrows between steps
            if i < len(steps) - 1:
                arrow = ConnectionPatch((50, step['y']-4), (50, steps[i+1]['y']+6), 
                                      "data", "data", arrowstyle="-|>", 
                                      shrinkA=0, shrinkB=0, mutation_scale=20, 
                                      fc=self.colors['text'], ec=self.colors['text'], linewidth=2)
                ax.add_patch(arrow)
        
        # Add side annotations - larger and better styled
        annotations = [
            {'text': 'Supports:\nEQDyna • FD3D\nWaveQLab3D • SeisSol', 'y': 78, 'side': 'left'},
            {'text': 'NPZ Format\nStandardization', 'y': 66, 'side': 'right'},
            {'text': 'Flexible Grid\nResolution Options', 'y': 54, 'side': 'left'},
            {'text': 'PGA • PGV • RSA\nComprehensive Metrics', 'y': 42, 'side': 'right'},
            {'text': 'Interpolated\nContour Mapping', 'y': 30, 'side': 'left'},
            {'text': 'Distance-based\nAttenuation Analysis', 'y': 18, 'side': 'right'},
            {'text': 'Interactive Web\nInterface & Reports', 'y': 6, 'side': 'left'}
        ]
        
        for ann in annotations:
            x_pos = 8 if ann['side'] == 'left' else 92
            # Create annotation boxes
            ann_box = FancyBboxPatch((x_pos-6, ann['y']-3), 12, 6, boxstyle="round,pad=0.5", 
                                   facecolor=self.colors['background'], alpha=0.8, 
                                   edgecolor=self.colors['info'], linewidth=1)
            ax.add_patch(ann_box)
            ax.text(x_pos, ann['y'], ann['text'], ha='center', va='center', 
                   fontsize=11, color=self.colors['text'], fontweight='normal')
            
            # Connect to main workflow
            if ann['side'] == 'left':
                connector = ConnectionPatch((x_pos+6, ann['y']), (35, ann['y']), "data", "data",
                                          arrowstyle="-", shrinkA=0, shrinkB=0, 
                                          fc=self.colors['info'], ec=self.colors['info'], 
                                          linewidth=1, alpha=0.5, linestyle='--')
            else:
                connector = ConnectionPatch((65, ann['y']), (x_pos-6, ann['y']), "data", "data",
                                          arrowstyle="-", shrinkA=0, shrinkB=0, 
                                          fc=self.colors['info'], ec=self.colors['info'], 
                                          linewidth=1, alpha=0.5, linestyle='--')
            ax.add_patch(connector)
        
        plt.tight_layout()
        return fig
    
    def create_business_overview(self):
        """Create business-focused overview"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        # Title
        title_box = FancyBboxPatch((10, 88), 80, 10, boxstyle="round,pad=1.5", 
                                  facecolor=self.colors['primary'], alpha=0.9, edgecolor='none')
        ax.add_patch(title_box)
        ax.text(50, 93, 'DR4GM Business Value Proposition', 
                ha='center', va='center', fontsize=22, fontweight='bold', color='white')
        
        # Problem vs Solution - side by side with better styling
        # Problem statement
        problem_box = FancyBboxPatch((5, 65), 40, 20, boxstyle="round,pad=1.5", 
                                   facecolor='#FEE2E2', edgecolor='#DC2626', linewidth=3)
        ax.add_patch(problem_box)
        ax.text(25, 80, 'CURRENT CHALLENGES', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#DC2626')
        
        problems = [
            '• Manual data processing takes weeks',
            '• Incompatible simulation formats',
            '• High error rates in analysis',
            '• Expensive specialized expertise required'
        ]
        for i, problem in enumerate(problems):
            ax.text(25, 76-i*2.5, problem, ha='center', va='center', fontsize=12, color='#7F1D1D')
        
        # Solution
        solution_box = FancyBboxPatch((55, 65), 40, 20, boxstyle="round,pad=1.5", 
                                    facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=3)
        ax.add_patch(solution_box)
        ax.text(75, 80, 'DR4GM SOLUTION', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#16A34A')
        
        solutions = [
            '• Automated pipeline: minutes to results',
            '• Universal format support',
            '• 100% reproducible accuracy',
            '• No specialized expertise needed'
        ]
        for i, solution in enumerate(solutions):
            ax.text(75, 76-i*2.5, solution, ha='center', va='center', fontsize=12, color='#14532D')
        
        # Arrow between problem and solution
        arrow = ConnectionPatch((45, 75), (55, 75), "data", "data",
                               arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=25, 
                               fc=self.colors['accent'], ec=self.colors['accent'], linewidth=4)
        ax.add_patch(arrow)
        ax.text(50, 77, 'DR4GM', ha='center', va='center', fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=self.colors['accent']))
        
        # ROI metrics - large prominent boxes
        roi_y = 55
        ax.text(50, roi_y, 'RETURN ON INVESTMENT', ha='center', va='center', 
               fontsize=18, fontweight='bold', color=self.colors['primary'])
        
        roi_data = [
            {'metric': '80%', 'label': 'Time Savings', 'desc': 'Weeks → Minutes', 'color': self.colors['light_blue'], 'border': self.colors['primary']},
            {'metric': '50%', 'label': 'Cost Reduction', 'desc': 'Eliminate Manual Work', 'color': self.colors['light_orange'], 'border': self.colors['accent']},
            {'metric': '100%', 'label': 'Accuracy', 'desc': 'Physics-Based Results', 'color': self.colors['light_green'], 'border': self.colors['success']}
        ]
        
        for i, roi in enumerate(roi_data):
            x_pos = 15 + i * 30
            roi_box = FancyBboxPatch((x_pos-8, 25), 16, 18, boxstyle="round,pad=1", 
                                   facecolor=roi['color'], edgecolor=roi['border'], linewidth=2)
            ax.add_patch(roi_box)
            
            # Large metric number
            ax.text(x_pos, 38, roi['metric'], ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=roi['border'])
            
            # Label
            ax.text(x_pos, 33, roi['label'], ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=roi['border'])
            
            # Description
            ax.text(x_pos, 29, roi['desc'], ha='center', va='center', 
                   fontsize=11, color=self.colors['text'])
        
        # Call to action - prominent bottom section
        cta_box = FancyBboxPatch((10, 5), 80, 15, boxstyle="round,pad=1.5", 
                               facecolor=self.colors['primary'], alpha=0.9, 
                               edgecolor=self.colors['primary'], linewidth=2)
        ax.add_patch(cta_box)
        ax.text(50, 15, 'Ready to Transform Your Earthquake Risk Analysis?', 
               ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        ax.text(50, 11, 'Reduce analysis time from weeks to minutes', 
               ha='center', va='center', fontsize=14, color='white')
        ax.text(50, 7, 'Contact us for a demonstration', 
               ha='center', va='center', fontsize=12, color='white', style='italic')
        
        plt.tight_layout()
        return fig
    
    def generate_diagram(self, output_path=None):
        """Generate the appropriate diagram based on style"""
        if self.style == 'comprehensive':
            fig = self.create_comprehensive_diagram()
        elif self.style == 'technical':
            fig = self.create_technical_flowchart()
        elif self.style == 'business':
            fig = self.create_business_overview()
        else:
            raise ValueError(f"Unknown style: {self.style}")
        
        if output_path is None:
            output_path = f"DR4GM_{self.style}_diagram.pdf"
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate DR4GM capability diagrams')
    parser.add_argument('--style', choices=['comprehensive', 'technical', 'business'], 
                       default='comprehensive', help='Diagram style to generate')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--all', action='store_true', help='Generate all diagram styles')
    
    args = parser.parse_args()
    
    if args.all:
        styles = ['comprehensive', 'technical', 'business']
        generated_files = []
        for style in styles:
            generator = DR4GMDiagramGenerator(style=style)
            output_path = generator.generate_diagram()
            generated_files.append(output_path)
            print(f"Generated {style} diagram: {output_path}")
        
        print(f"\nAll diagrams generated: {', '.join(generated_files)}")
    else:
        generator = DR4GMDiagramGenerator(style=args.style)
        output_path = generator.generate_diagram(args.output)
        print(f"Generated {args.style} diagram: {output_path}")

if __name__ == "__main__":
    main()