# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np
# import pandas as pd
# import streamlit 

# class FlowCurve():
#     def __init__(self):
#         self.theoretical_flow = None
#         self.theoretical_pressure = None
#         self.theoretical_data = None
        
#         self.n_actual_points = 0

#         self.desired_pressure = 0
#         self.desired_flow = 0
#         self.desired_height = 0
#         pass
    
#     def calculate_theoretical_curve(self, theoretical_flow, theoretical_pressure):
#         self.theoretical_flow = theoretical_flow
#         self.theoretical_pressure = theoretical_pressure
#         self.theoretical_data=pd.DataFrame({    
#                                 "x":[0, theoretical_flow, theoretical_flow*1.5], 
#                                 "y":[theoretical_pressure*1.2, theoretical_pressure, theoretical_pressure*0.65]
#                                })

#     def plot_data(self):
#         fig = go.Figure()
#         fig.update_layout(
#             title='N1.85 Pump Characteristic Curve',
#             xaxis_title='Flow Rate (log scale)',
#             yaxis_title='Pressure',
#             xaxis_type='log',  # Logarithmic x-axis
#             template='plotly_white',
#             # range_x=[1, 10000]
#         )
#         if self.theoretical_data is not None:
#             fig.add_trace(go.Scatter(x=self.theoretical_data["x"], y=self.theoretical_data["y"], mode='lines+markers', name='Theoretical Data'))
#         fig.update(layout_xaxis_range = [1,5])
#         fig.show()

# if __name__ == "__main__":
#     flow_curve = FlowCurve()
#     flow_curve.calculate_theoretical_curve(1000, 100)
#     flow_curve.plot_data()
import plotly.graph_objects as go
import numpy as np
import pandas as pd

class FlowCurve:
    def __init__(self):
        self.theoretical_flow = None
        self.theoretical_pressure = None
        self.theoretical_data = None

    def generate_n185_ticks(self, max_value):
        """
        Generate tick locations using N1.85 characteristic
        
        Args:
            max_value (float): Maximum flow rate
        
        Returns:
            List of tick locations
        """
        # Create a sequence of ticks with N1.85 spacing
        base_ticks = [0, 10, 100, 1000, 10000]
        
        # Filter ticks less than max_value
        ticks = [tick for tick in base_ticks if tick <= max_value]
        
        return ticks

    def calculate_theoretical_curve(self, theoretical_flow, theoretical_pressure):
        """
        Calculate theoretical curve points
        
        Args:
            theoretical_flow (float): Design point flow rate
            theoretical_pressure (float): Design point pressure
        """
        self.theoretical_flow = theoretical_flow
        self.theoretical_pressure = theoretical_pressure
        
        # Generate points for the curve
        x_points = [0, theoretical_flow, theoretical_flow*1.5]
        y_points = [
            theoretical_pressure*1.2,  # Starting point
            theoretical_pressure,      # Design point 
            theoretical_pressure*0.65  # Extended point
        ]
        
        self.theoretical_data = pd.DataFrame({
            "x": x_points,
            "y": y_points
        })

    def plot_data(self):
        """
        Create a Plotly figure for the flow curve with N1.85 tick spacing
        """
        fig = go.Figure()
        
        # Layout configuration
        fig.update_layout(
            title='N1.85 Pump Characteristic Curve',
            xaxis_title='Flow Rate (GPM)',
            yaxis_title='Pressure (PSI)',
            template='plotly_white',
            width=800,
            height=600,
            xaxis=dict(
                type='log',  # Use log scale
                range=[0, np.log10(max(self.theoretical_data['x'])*1000)]  # Dynamic range
            )
        )
        
        # Generate appropriate ticks
        ticks = self.generate_n185_ticks(max(self.theoretical_data['x'])*1000)
        
        # Update x-axis with custom ticks
        fig.update_xaxes(
            tickmode='array',
            tickvals=ticks,
            ticktext=[str(tick) for tick in ticks]
        )
        
        # Add theoretical data trace
        if self.theoretical_data is not None:
            fig.add_trace(go.Scatter(
                x=self.theoretical_data["x"], 
                y=self.theoretical_data["y"], 
                mode='lines+markers', 
                name='Theoretical Curve'
            ))
        
        return fig

def main():
    # Example usage
    flow_curve = FlowCurve()
    
    # Calculate theoretical curve
    flow_curve.calculate_theoretical_curve(1000, 100)
    
    # Create and show plot
    fig = flow_curve.plot_data()
    fig.show()

if __name__ == "__main__":
    main()