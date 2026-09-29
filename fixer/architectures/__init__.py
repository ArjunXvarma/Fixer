from fixer.architectures import autonomous_agent, react, simple_planner

ARCHITECTURES = {
    "react": react.build,
    "simple_planner": simple_planner.build,
    "autonomous_agent": autonomous_agent.build,
}
