#!/usr/bin/env python3
"""
PowerPoint-Style DR4GM Capability Diagram Generator

Creates presentation-ready diagrams that look like professional PowerPoint slides
with clean layouts, large fonts, and minimal content.

Usage:
    python generate_powerpoint_style_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

class PowerPointStyleDiagramGenerator:
    def __init__(self):
        # Professional PowerPoint color scheme
        self.colors = {
            'primary_blue': '#0F4C75',      # Deep professional blue
            'accent_orange': '#FF6B35',     # Vibrant orange
            'success_green': '#2E8B57',     # Sea green
            'light_blue': '#E3F2FD',       # Very light blue
            'light_orange': '#FFF3E0',     # Very light orange  
            'light_green': '#E8F5E8',      # Very light green
            'dark_gray': '#2C3E50',        # Professional dark gray
            'light_gray': '#F5F5F5',       # Background gray
            'white': '#FFFFFF'
        }
        
        # Set PowerPoint-like styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 14,
            'font.weight': 'normal'
        })
    
    def create_title_slide(self):
        """Create a PowerPoint-style title slide"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))  # 16:9 aspect ratio like PowerPoint
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis('off')
        fig.patch.set_facecolor(self.colors['white'])
        
        # Background gradient effect (simplified)
        gradient = Rectangle((0, 0), 16, 9, facecolor=self.colors['light_gray'], alpha=0.3)
        ax.add_patch(gradient)
        
        # Main title - very large and bold
        ax.text(8, 6.5, 'DR4GM', ha='center', va='center', 
               fontsize=64, fontweight='bold', color=self.colors['primary_blue'])
        
        # Subtitle
        ax.text(8, 5.5, 'Earthquake Simulation Processing Platform', 
                ha='center', va='center', fontsize=24, 
                color=self.colors['dark_gray'], fontweight='normal')
        
        # Value proposition
        ax.text(8, 4.5, 'Transform weeks of analysis into minutes of insight', 
                ha='center', va='center', fontsize=18, 
                color=self.colors['accent_orange'], style='italic')
        
        # Key stats - three prominent boxes
        stats_y = 3
        box_width = 4
        box_height = 1.2
        
        # Speed box
        speed_box = FancyBboxPatch((1, stats_y-0.6), box_width, box_height, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=self.colors['light_blue'], 
                                  edgecolor=self.colors['primary_blue'], linewidth=2)
        ax.add_patch(speed_box)
        ax.text(3, stats_y, '80% Faster', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['primary_blue'])
        ax.text(3, stats_y-0.3, 'Processing Time', ha='center', va='center', 
               fontsize=12, color=self.colors['dark_gray'])
        
        # Accuracy box
        accuracy_box = FancyBboxPatch((6, stats_y-0.6), box_width, box_height, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=self.colors['light_orange'], 
                                     edgecolor=self.colors['accent_orange'], linewidth=2)
        ax.add_patch(accuracy_box)
        ax.text(8, stats_y, '100% Accurate', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['accent_orange'])
        ax.text(8, stats_y-0.3, 'Physics-Based', ha='center', va='center', 
               fontsize=12, color=self.colors['dark_gray'])
        
        # Formats box
        formats_box = FancyBboxPatch((11, stats_y-0.6), box_width, box_height, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=self.colors['light_green'], 
                                    edgecolor=self.colors['success_green'], linewidth=2)
        ax.add_patch(formats_box)
        ax.text(13, stats_y, '4+ Formats', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['success_green'])
        ax.text(13, stats_y-0.3, 'Supported', ha='center', va='center', 
               fontsize=12, color=self.colors['dark_gray'])
        
        # Footer
        ax.text(8, 0.5, 'For Emergency Managers • Research Scientists • Engineering Professionals', 
                ha='center', va='center', fontsize=14, 
                color=self.colors['dark_gray'], alpha=0.8)
        
        plt.tight_layout()
        return fig
    
    def create_workflow_slide(self):
        """Create a clean 3-step workflow slide"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis('off')
        fig.patch.set_facecolor(self.colors['white'])
        
        # Title
        ax.text(8, 8.2, 'How DR4GM Works', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=self.colors['primary_blue'])
        
        # Three large workflow boxes
        box_y = 5
        box_width = 4
        box_height = 2.5
        
        # Step 1: INPUT
        step1_box = FancyBboxPatch((1, box_y-1.25), box_width, box_height, 
                                  boxstyle="round,pad=0.2", 
                                  facecolor=self.colors['primary_blue'], 
                                  edgecolor='none')
        ax.add_patch(step1_box)
        ax.text(3, box_y+0.5, '1', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=self.colors['white'])
        ax.text(3, box_y, 'INPUT', ha='center', va='center', 
               fontsize=24, fontweight='bold', color=self.colors['white'])
        ax.text(3, box_y-0.5, 'Raw Simulation Data', ha='center', va='center', 
               fontsize=14, color=self.colors['white'])
        ax.text(3, box_y-0.8, 'EQDyna • FD3D • WaveQLab3D', ha='center', va='center', 
               fontsize=12, color=self.colors['white'])
        
        # Step 2: PROCESS
        step2_box = FancyBboxPatch((6, box_y-1.25), box_width, box_height, 
                                  boxstyle="round,pad=0.2", 
                                  facecolor=self.colors['accent_orange'], 
                                  edgecolor='none')
        ax.add_patch(step2_box)
        ax.text(8, box_y+0.5, '2', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=self.colors['white'])
        ax.text(8, box_y, 'PROCESS', ha='center', va='center', 
               fontsize=24, fontweight='bold', color=self.colors['white'])
        ax.text(8, box_y-0.5, 'Automated Pipeline', ha='center', va='center', 
               fontsize=14, color=self.colors['white'])
        ax.text(8, box_y-0.8, 'Convert • Analyze • Visualize', ha='center', va='center', 
               fontsize=12, color=self.colors['white'])
        
        # Step 3: RESULTS
        step3_box = FancyBboxPatch((11, box_y-1.25), box_width, box_height, 
                                  boxstyle="round,pad=0.2", 
                                  facecolor=self.colors['success_green'], 
                                  edgecolor='none')
        ax.add_patch(step3_box)
        ax.text(13, box_y+0.5, '3', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=self.colors['white'])
        ax.text(13, box_y, 'RESULTS', ha='center', va='center', 
               fontsize=24, fontweight='bold', color=self.colors['white'])
        ax.text(13, box_y-0.5, 'Decision-Ready', ha='center', va='center', 
               fontsize=14, color=self.colors['white'])
        ax.text(13, box_y-0.8, 'Maps • Stats • Reports', ha='center', va='center', 
               fontsize=12, color=self.colors['white'])
        
        # Large arrows between steps
        # Arrow 1 -> 2
        ax.annotate('', xy=(5.8, box_y), xytext=(5.2, box_y),
                   arrowprops=dict(arrowstyle='->', lw=4, color=self.colors['dark_gray']))
        
        # Arrow 2 -> 3
        ax.annotate('', xy=(10.8, box_y), xytext=(10.2, box_y),
                   arrowprops=dict(arrowstyle='->', lw=4, color=self.colors['dark_gray']))
        
        # Time comparison at bottom
        time_y = 2.5
        
        # Before DR4GM
        before_box = FancyBboxPatch((2, time_y-0.4), 5, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFEBEE', 
                                   edgecolor='#D32F2F', linewidth=2)
        ax.add_patch(before_box)
        ax.text(4.5, time_y, 'Before DR4GM: WEEKS', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#D32F2F')
        
        # After DR4GM
        after_box = FancyBboxPatch((9, time_y-0.4), 5, 0.8, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E8F5E8', 
                                  edgecolor='#2E8B57', linewidth=2)
        ax.add_patch(after_box)
        ax.text(11.5, time_y, 'With DR4GM: MINUTES', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#2E8B57')
        
        # VS in between
        ax.text(8, time_y, 'VS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['dark_gray'])
        
        plt.tight_layout()
        return fig
    
    def create_benefits_slide(self):
        """Create a benefits overview slide"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis('off')
        fig.patch.set_facecolor(self.colors['white'])
        
        # Title
        ax.text(8, 8.2, 'Who Benefits from DR4GM?', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=self.colors['primary_blue'])
        
        # Three user groups - large boxes
        box_y = 5.5
        box_width = 4.5
        box_height = 3
        
        # Emergency Managers
        em_box = FancyBboxPatch((0.75, box_y-1.5), box_width, box_height, 
                               boxstyle="round,pad=0.2", 
                               facecolor=self.colors['light_blue'], 
                               edgecolor=self.colors['primary_blue'], linewidth=3)
        ax.add_patch(em_box)
        ax.text(3, box_y+0.8, 'EMERGENCY', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['primary_blue'])
        ax.text(3, box_y+0.4, 'MANAGERS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['primary_blue'])
        
        em_benefits = [
            '• Risk Assessment', 
            '• Decision Support', 
            '• Resource Planning',
            '• Real-time Analysis'
        ]
        for i, benefit in enumerate(em_benefits):
            ax.text(3, box_y-0.2-i*0.3, benefit, ha='center', va='center', 
                   fontsize=14, color=self.colors['dark_gray'])
        
        # Research Scientists
        rs_box = FancyBboxPatch((5.75, box_y-1.5), box_width, box_height, 
                               boxstyle="round,pad=0.2", 
                               facecolor=self.colors['light_orange'], 
                               edgecolor=self.colors['accent_orange'], linewidth=3)
        ax.add_patch(rs_box)
        ax.text(8, box_y+0.8, 'RESEARCH', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['accent_orange'])
        ax.text(8, box_y+0.4, 'SCIENTISTS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['accent_orange'])
        
        rs_benefits = [
            '• Ground Motion Metrics', 
            '• Statistical Analysis', 
            '• Data Standards',
            '• Publication Quality'
        ]
        for i, benefit in enumerate(rs_benefits):
            ax.text(8, box_y-0.2-i*0.3, benefit, ha='center', va='center', 
                   fontsize=14, color=self.colors['dark_gray'])
        
        # Engineers
        eng_box = FancyBboxPatch((10.75, box_y-1.5), box_width, box_height, 
                                boxstyle="round,pad=0.2", 
                                facecolor=self.colors['light_green'], 
                                edgecolor=self.colors['success_green'], linewidth=3)
        ax.add_patch(eng_box)
        ax.text(13, box_y+0.8, 'ENGINEERING', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['success_green'])
        ax.text(13, box_y+0.4, 'PROFESSIONALS', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['success_green'])
        
        eng_benefits = [
            '• Design Support', 
            '• Site-Specific Data', 
            '• Code Compliance',
            '• Hazard Assessment'
        ]
        for i, benefit in enumerate(eng_benefits):
            ax.text(13, box_y-0.2-i*0.3, benefit, ha='center', va='center', 
                   fontsize=14, color=self.colors['dark_gray'])
        
        # Bottom call to action
        cta_box = FancyBboxPatch((2, 1), 12, 1, boxstyle="round,pad=0.1", 
                               facecolor=self.colors['dark_gray'], edgecolor='none')
        ax.add_patch(cta_box)
        ax.text(8, 1.5, 'Ready to transform your earthquake risk analysis?', 
                ha='center', va='center', fontsize=20, fontweight='bold', color=self.colors['white'])
        
        plt.tight_layout()
        return fig
    
    def generate_all_slides(self):
        """Generate all PowerPoint-style slides"""
        slides = []
        
        # Generate title slide
        fig1 = self.create_title_slide()
        fig1.savefig('DR4GM_Title_Slide.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        slides.append('DR4GM_Title_Slide.png')
        plt.close(fig1)
        
        # Generate workflow slide
        fig2 = self.create_workflow_slide()
        fig2.savefig('DR4GM_Workflow_Slide.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        slides.append('DR4GM_Workflow_Slide.png')
        plt.close(fig2)
        
        # Generate benefits slide
        fig3 = self.create_benefits_slide()
        fig3.savefig('DR4GM_Benefits_Slide.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        slides.append('DR4GM_Benefits_Slide.png')
        plt.close(fig3)
        
        return slides

def main():
    generator = PowerPointStyleDiagramGenerator()
    slides = generator.generate_all_slides()
    
    print("Generated PowerPoint-style slides:")
    for slide in slides:
        print(f"  - {slide}")
    
    print("\nTo use these slides:")
    print("1. Insert these images into PowerPoint slides")
    print("2. Each image is optimized for 16:9 slide format")
    print("3. High resolution (300 DPI) suitable for projection")
    print("4. Clean, professional design with large fonts")

if __name__ == "__main__":
    main()