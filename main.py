import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

def calculate_theoretical_curve(flow, pressure):
    """Calculate theoretical curve with two linear segments:
    1. From (0, pressure*1.2) to (flow, pressure)
    2. From (flow, pressure) to (flow*2.5, 0)
    """
    # First segment: (0, pressure*1.2) to (flow, pressure)
    x1 = np.array([0, flow])
    y1 = np.array([pressure * 1.2, pressure])
    
    # Second segment: (flow, pressure) to (flow*2.5, 0)
    x2 = np.array([flow, flow * 2.5])
    y2 = np.array([pressure, 0])
    
    # Find where the second line intersects y=0 to extend it
    if pressure > 0:  # Avoid division by zero
        m = (y2[1] - y2[0]) / (x2[1] - x2[0])  # slope
        b = y2[0] - m * x2[0]  # y-intercept
        x_zero = -b / m if m != 0 else x2[1]  # x where y=0
        
        # If x_zero is beyond flow*2.5, extend the line
        if x_zero > flow * 2.5:
            x2 = np.array([flow, x_zero])
            y2 = np.array([pressure, 0])
    
    # Combine the segments
    x = np.concatenate([x1, x2[1:]])  # Skip the first point of x2 to avoid duplication
    y = np.concatenate([y1, y2[1:]])  # Skip the first point of y2 to avoid duplication
    
    return x, y

def extend_line_for_intersection(x1, y1, x2, y2, max_flow=3000, max_pressure=3000):
    """
    Extend a line defined by two points to intersect with x-axis and other bounds
    """
    if x2 - x1 == 0:  # Vertical line
        return np.array([x1, x1]), np.array([0, max_pressure])
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    x_axis_intersect = -b/m if m != 0 else max_flow
    
    x_min = 0
    x_max = min(max_flow, x_axis_intersect if x_axis_intersect > 0 else max_flow)
    
    x_extended = np.linspace(x_min, x_max, 1000)
    y_extended = m * x_extended + b
    
    y_extended = np.clip(y_extended, 0, max_pressure)
    
    mask = y_extended > 0.1
    return x_extended[mask], y_extended[mask]

def process_pump_curve(pump_df, max_flow=3000):
    """Process pump curve data and extend to x-axis"""
    if pump_df.empty:
        return None, None
    
    pump_df = pump_df.sort_values('x').dropna()
    if len(pump_df) < 2:
        return None, None
    
    try:
        x = pump_df['x'].values
        y = pump_df['y'].values
        f = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        
        x_extended = np.linspace(0, max_flow, 1000)
        y_extended = f(x_extended)
        
        positive_mask = y_extended > 0
        x_extended = x_extended[positive_mask]
        y_extended = y_extended[positive_mask]
        
        return x_extended, y_extended
    except Exception as e:
        st.error(f"Error processing pump curve: {str(e)}")
        return None, None

def find_intersections(x1, y1, x2, y2):
    """Find intersection points between two curves"""
    try:
        f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
        f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')
        
        x_min = max(min(x1), min(x2))
        x_max = min(max(x1), max(x2))
        x_test = np.linspace(x_min, x_max, 1000)
        
        y1_test = f1(x_test)
        y2_test = f2(x_test)
        y_diff = y1_test - y2_test
        
        zero_crossings = np.where(np.diff(np.signbit(y_diff)))[0]
        
        intersect_x = []
        intersect_y = []
        for i in zero_crossings:
            x_int = (x_test[i] + x_test[i + 1]) / 2
            y_int = f1(x_int)
            intersect_x.append(x_int)
            intersect_y.append(y_int)
            
        return intersect_x, intersect_y
    except Exception as e:
        st.error(f"Error finding intersections: {str(e)}")
        return [], []

def process_actual_curve(x1, y1, x2, y2, theo_x, theo_y, pump_x, pump_y, max_flow=3000, max_pressure=3000):
    """
    Process actual curve data and find intersections with other curves.
    Returns the trimmed actual curve and all intersection points.
    """
    # Get extended line
    actual_x, actual_y = extend_line_for_intersection(x1, y1, x2, y2, max_flow, max_pressure)
    
    all_intersections = []
    
    # Find intersections with theoretical curve
    x_int, y_int = find_intersections(theo_x, theo_y, actual_x, actual_y)
    if x_int:
        all_intersections.extend(zip(x_int, y_int))
    
    # Find intersections with pump curve if it exists
    if pump_x is not None and pump_y is not None:
        x_int, y_int = find_intersections(pump_x, pump_y, actual_x, actual_y)
        if x_int:
            all_intersections.extend(zip(x_int, y_int))
    
    # If intersections found, trim the actual curve
    if all_intersections:
        # Sort intersections by x value
        all_intersections.sort(key=lambda p: p[0])
        last_intersection = all_intersections[-1]
        
        # Trim actual curve to last intersection point
        mask = actual_x <= last_intersection[0]
        actual_x = actual_x[mask]
        actual_y = actual_y[mask]
        
        # Add the last intersection point to ensure the curve ends exactly there
        actual_x = np.append(actual_x, last_intersection[0])
        actual_y = np.append(actual_y, last_intersection[1])
    
    return actual_x, actual_y, all_intersections

def main():
    st.title("Fire Protection System Design: Pump Curve Analysis")

    # Initialize session state for theoretical values and plot trigger
    if 'theoretical_flow' not in st.session_state:
        st.session_state.theoretical_flow = 1000.0
    if 'theoretical_pressure' not in st.session_state:
        st.session_state.theoretical_pressure = 130.0
    if 'plot_trigger' not in st.session_state:
        st.session_state.plot_trigger = False

    # Top section: 3 columns
    col1, col2, col3 = st.columns(3)

    # Column 1: Theoretical values
    with col1:
        st.header("Theoretical Pump Curve")
        theoretical_flow = st.number_input(
            "Flow", 
            min_value=0.0, 
            value=st.session_state.theoretical_flow,
            key="flow_input"
        )
        theoretical_pressure = st.number_input(
            "Pressure", 
            min_value=0.0, 
            value=st.session_state.theoretical_pressure,
            key="pressure_input"
        )
        
        st.session_state.theoretical_flow = theoretical_flow
        st.session_state.theoretical_pressure = theoretical_pressure

    # Column 2: Actual values
    with col2:
        st.header("Sprinkler Demand")
        # Create a base DataFrame with no index displayed
        actual_df = st.data_editor(
            pd.DataFrame({'Flow Rate': [0.0, 0.0], 'Pressure': [0.0, 0.0]}),
            num_rows=2,
            hide_index=True
        )
        # Rename columns back to x,y for compatibility with existing code
        actual_df.columns = ['x', 'y']

    # Column 3: Pump values
    with col3:
        st.header("Actual Pump Curve")
        pump_df = st.data_editor(
            pd.DataFrame({'Flow Rate': [], 'Pressure': []}),
            num_rows="dynamic",
            hide_index=True
        )
        # Convert to proper column names and ensure numeric types
        pump_df.columns = ['x', 'y']
        pump_df = pump_df.apply(pd.to_numeric, errors='coerce')

    # Submit button in a single column
    submit_button = st.button("Generate Plot")

    # Only plot when submit button is clicked
    if submit_button:
        try:
            # Create figure
            fig = go.Figure()
            
            # 1. Process theoretical curve
            theo_x, theo_y = calculate_theoretical_curve(theoretical_flow, theoretical_pressure)
            fig.add_trace(go.Scatter(x=theo_x, y=theo_y, mode='lines', name='Theoretical'))
            
            # Add annotation for theoretical point
            fig.add_annotation(
                x=theoretical_flow,
                y=theoretical_pressure,
                text=f"({theoretical_flow:.1f}, {theoretical_pressure:.1f})",
                showarrow=True,
                arrowhead=1,
                yshift=4,
                # bordercolor="white",
                borderwidth=1
            )
            
            # 2. Process pump curve
            pump_x, pump_y = process_pump_curve(pump_df)
            if pump_x is not None and pump_y is not None:
                fig.add_trace(go.Scatter(x=pump_x, y=pump_y, mode='lines', name='Pump'))
            
            # 3. Process actual curve and find intersections
            actual_df = actual_df.dropna()
            if len(actual_df) == 2:
                actual_x, actual_y, all_intersections = process_actual_curve(
                    actual_df['x'].iloc[0], actual_df['y'].iloc[0],
                    actual_df['x'].iloc[1], actual_df['y'].iloc[1],
                    theo_x, theo_y, pump_x, pump_y
                )
                fig.add_trace(go.Scatter(
                        x=[actual_df['x'].iloc[0]], y=[actual_df['y'].iloc[0]],
                        mode='markers',
                        name='Set Point',
                        marker=dict(size=10, color='red'),
                        hoverinfo='text'
                    ))
                fig.add_annotation(
                            x=actual_df['x'].iloc[0],
                            y=actual_df['y'].iloc[0],
                            text=f"({actual_df['x'].iloc[0]:.1f}, {actual_df['y'].iloc[0]:.1f})",
                            showarrow=True,
                            arrowhead=1,
                            yshift=4,
                            # bordercolor="white",
                            borderwidth=1
                        )
                fig.add_trace(go.Scatter(
                        x=[actual_df['x'].iloc[1]], y=[actual_df['y'].iloc[1]],
                        mode='markers',
                        name='Set Point',
                        marker=dict(size=10, color='red'),
                        # text=[f"({x:.2f}, {y:.2f})" for x, y in zip(actual_df['x'].iloc[1], actual_df['y'].iloc[1])],
                        hoverinfo='text'
                    ))
                fig.add_annotation(
                            x=actual_df['x'].iloc[1],
                            y=actual_df['y'].iloc[1],
                            text=f"({actual_df['x'].iloc[1]:.1f}, {actual_df['y'].iloc[1]:.1f})",
                            showarrow=True,
                            arrowhead=1,
                            yshift=4,
                            # bordercolor="white",
                            borderwidth=1
                        )
                
                # Plot trimmed actual curve
                fig.add_trace(go.Scatter(x=actual_x, y=actual_y, mode='lines', name='Actual'))
                
                # Find intersections between Theoretical and Pump
                if pump_x is not None:
                    x_int, y_int = find_intersections(theo_x, theo_y, pump_x, pump_y)
                    if x_int:
                        all_intersections.extend(zip(x_int, y_int))
                
                # Plot intersection points and add annotations
                if all_intersections:
                    intersect_x, intersect_y = zip(*all_intersections)
                    fig.add_trace(go.Scatter(
                        x=intersect_x, y=intersect_y,
                        mode='markers',
                        name='Intersections',
                        marker=dict(size=10, color='red'),
                        text=[f"({x:.2f}, {y:.2f})" for x, y in zip(intersect_x, intersect_y)],
                        hoverinfo='text'
                    ))
                    
                    # Add annotations for intersection points
                    for x, y in zip(intersect_x, intersect_y):
                        fig.add_annotation(
                            x=x,
                            y=y,
                            text=f"({x:.1f}, {y:.1f})",
                            showarrow=True,
                            arrowhead=1,
                            yshift=4,
                            # bordercolor="white",
                            borderwidth=1
                        )
                    
                    # Display intersection points in a table
                    st.subheader("Intersection Points")
                    intersections_df = pd.DataFrame(all_intersections, columns=['Flow', 'Pressure'])
                    st.dataframe(intersections_df)

            # Update layout
            fig.update_layout(
                xaxis_title="Flow",
                yaxis_title="Pressure",
                title="Pump Curve Analysis",
                showlegend=True,
                width=800,
                height=600
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")
            st.write("Debug info:", str(e))

if __name__ == "__main__":
    main()