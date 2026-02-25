"""Gradio UI Service for Energy Advisor."""

import os
import gradio as gr
import httpx

# Route traffic to the API Gateway container via Docker's internal DNS
API_URL = os.getenv("API_URL", "http://api:8000")

def analyze_building(
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    overall_height,
    orientation,
    glazing_area,
    glazing_distribution
):
    """Call API gateway for two-stage analysis."""

    # Map human-readable UI inputs to the strict Pydantic integer constraints
    orientation_map = {"North": 2, "East": 3, "South": 4, "West": 5}
    distribution_map = {"Unknown": 0, "Uniform": 1, "North": 2, "East": 3, "South": 4, "West": 5}

    payload = {
        "relative_compactness": relative_compactness,
        "surface_area": surface_area,
        "wall_area": wall_area,
        "roof_area": roof_area,
        "overall_height": overall_height,
        "orientation": orientation_map.get(orientation, 4),
        "glazing_area": glazing_area,
        "glazing_area_distribution": distribution_map.get(glazing_distribution, 0)
    }

    try:
        # 60s timeout to accommodate SLM generation
        response = httpx.post(f"{API_URL}/analyze", json=payload, timeout=60.0)

        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"

        result = response.json()

        output = f"""## 🏢 Energy Efficiency Analysis

### 📊 Predicted Energy Loads (XGBoost)
- **Heating Load:** {result['heating_load']:.1f} kWh/m²
- **Cooling Load:** {result['cooling_load']:.1f} kWh/m²
- **Efficiency Score:** {result['efficiency_score']:.0f}/100
- **Confidence:** {result['confidence']*100:.0f}%

### 💡 Recommendations (SLM-Generated)
"""

        for i, rec in enumerate(result.get('recommendations', []), 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
            output += f"\n{i}. {priority_emoji} **{rec['category'].title()}** ({rec['priority']} priority)\n"
            output += f"   - Action: {rec['action']}\n"
            output += f"   - Expected Impact: {rec['expected_impact']}\n"

        output += f"\n### 📝 Analysis\n{result['explanation']}\n"
        output += f"\n*Pipeline Version: {result.get('model_version', 'Unknown')}*"

        return output

    except httpx.RequestError as e:
        return f"❌ Network Error connecting to API Gateway: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

def create_ui():
    """Create Gradio interface."""
    with gr.Blocks(title="Energy Advisor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏢 Energy-Efficient Building Design Advisor
        **Two-Stage Pipeline:** XGBoost (Numerical Predictions) + TinyLlama (Actionable Explanations)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Building Specifications")

                # Inputs strictly match schema boundaries
                relative_compactness = gr.Slider(0.0, 1.0, 0.98, label="Relative Compactness")
                surface_area = gr.Number(514.5, label="Surface Area (m²)")
                wall_area = gr.Number(294.0, label="Wall Area (m²)")
                roof_area = gr.Number(110.25, label="Roof Area (m²)")
                overall_height = gr.Number(7.0, label="Overall Height (m)")
                orientation = gr.Dropdown(["North", "East", "South", "West"], value="South", label="Orientation")
                glazing_area = gr.Slider(0.0, 0.5, 0.1, label="Glazing Area Ratio")
                glazing_distribution = gr.Dropdown(
                    ["Unknown", "Uniform", "North", "East", "South", "West"],
                    value="Uniform",
                    label="Glazing Distribution"
                )

                submit_btn = gr.Button("Analyze Building", variant="primary")

            with gr.Column(scale=1):
                output = gr.Markdown("Enter specifications and click Analyze to generate efficiency recommendations.")

        submit_btn.click(
            fn=analyze_building,
            inputs=[relative_compactness, surface_area, wall_area, roof_area,
                   overall_height, orientation, glazing_area, glazing_distribution],
            outputs=output
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("UI_PORT", 7860)),
        share=False
    )