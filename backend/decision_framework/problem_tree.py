# backend/decision_framework/problem_tree.py
class ProblemNode:
    def __init__(self, title, description="", children=None):
        self.title = title
        self.description = description
        self.children = children or []

    def add_child(self, node):
        self.children.append(node)

    def to_dict(self):
        return {"title": self.title, "description": self.description,
                "children": [c.to_dict() for c in self.children]}
