"""
Every architecture is a module with one function: build() -> compiled graph.

To add one, drop a new module in this package and list it here.
"""

from fixer.architectures import autonomous_agent, react, simple_planner

ARCHITECTURES = {
    "react": react.build,
    "simple_planner": simple_planner.build,
    "autonomous_agent": autonomous_agent.build,
}
